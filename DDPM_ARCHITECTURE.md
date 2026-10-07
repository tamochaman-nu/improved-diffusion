# 学習中DDPMのアーキテクチャ仕様

CycleDiffusionへの組み込みを想定した、現在`logs/anime-aligned-curated`で学習中のDDPMの詳細仕様。
コードベースは`openai/improved-diffusion`(fork、変更点は[CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)参照)。

**現在の学習状況(2026-09-03時点)**: 無条件(class_cond=False)、256px、step 136,660+ (`lr_anneal_steps=400000`まで継続中)。
チェックポイントは`logs/anime-aligned-curated/{model,ema_0.9999,opt}{step:06d}.pt`([run_train.sh](run_train.sh))。

---

## 1. モデルを再構築するためのハイパーパラメータ

チェックポイント(`.pt`)は`state_dict`(テンソルの重みのみ)で、アーキテクチャ情報は一切含まれない。
以下のflagsで`create_model_and_diffusion()`([improved_diffusion/script_util.py:38](improved_diffusion/script_util.py#L38))を呼び、
`model.load_state_dict(...)`する必要がある。

```python
from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion

flags = model_and_diffusion_defaults()
flags.update(dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    attention_resolutions="32,16,8",   # -> ds {8,16,32} (image_size // res)
    learn_sigma=True,
    class_cond=False,
    diffusion_steps=1000,
    noise_schedule="linear",
    # 以下はデフォルト値のまま(run_train.shで明示していないため):
    # num_heads=4, num_heads_upsample=-1, dropout=0.0,
    # use_scale_shift_norm=True, predict_xstart=False, use_kl=False,
    # rescale_timesteps=True, rescale_learned_sigmas=True,
    # sigma_small=False, timestep_respacing="" (=full 1000 steps)
))
model, diffusion = create_model_and_diffusion(**flags)
model.load_state_dict(torch.load("model130000.pt", map_location="cpu"))  # or ema_0.9999_*.pt
```

`use_checkpoint`(gradient checkpointing)と`use_fp16`は学習時のメモリ最適化フラグで、
アーキテクチャ(重みの形状)には影響しない。推論では両方Falseで問題ない。

---

## 2. UNetバックボーン ([improved_diffusion/unet.py:283](improved_diffusion/unet.py#L283) `UNetModel`)

`image_size=256`により`channel_mult=(1,1,2,2,4,4)`が選ばれる
([script_util.py:99-106](improved_diffusion/script_util.py#L99-L106)。**256/64/32以外の`image_size`は`ValueError`になる点に注意**)。

`model_channels=256`, `num_res_blocks=2` として、エンコーダ側(`input_blocks`)は以下の6レベル・18ブロック構成:

| level | mult | channels | 解像度 | ds(downsample factor) | attention | 備考 |
|---|---|---|---|---|---|---|
| 0 | 1 | 256 | 256x256 | 1 | – | ResBlock x2 → Downsample |
| 1 | 1 | 256 | 128x128 | 2 | – | ResBlock x2 → Downsample |
| 2 | 2 | 512 | 64x64 | 4 | – | ResBlock x2 → Downsample |
| 3 | 2 | 512 | 32x32 | 8 | ✓ | ResBlock+Attn x2 → Downsample |
| 4 | 4 | 1024 | 16x16 | 16 | ✓ | ResBlock+Attn x2 → Downsample |
| 5 | 4 | 1024 | 8x8 | 32 | ✓ | ResBlock+Attn x2 (Downsampleなし) |

- attentionは`ds ∈ {8,16,32}`(`attention_resolutions="32,16,8"`を`image_size//res`に変換した値)のレベルにのみ挿入される
  ([unet.py:375](improved_diffusion/unet.py#L375))。
- **middle_block**(ボトルネック, 8x8, 1024ch): `ResBlock → AttentionBlock → ResBlock`。
- **output_blocks (デコーダ)**: 上記の逆順(level 5→0)で、各レベル`num_res_blocks+1=3`ブロック
  (対応する`input_blocks`の出力をchannel方向にconcatするため+1)、計18ブロック。
  `level != 0`のレベルの最終ブロックにUpsampleが付く(最終出力は256x256に戻る)。
- 入力: `conv_nd(2, 3, 256, kernel=3, padding=1)` (ステム畳み込み)。
- 出力: `GroupNorm32(32,256) → SiLU → zero-initialized Conv(256, 6, kernel=3)`
  (`out_channels=6`は`learn_sigma=True`のため、詳細は§4)。

### ResBlock ([unet.py:105](improved_diffusion/unet.py#L105))
```
h = Conv3x3(SiLU(GroupNorm32(x)))                      # in_layers
emb_out = Linear(SiLU(timestep_emb))                    # emb_layers, out=2*out_ch (scale-shift用)
scale, shift = chunk(emb_out, 2)
h = GroupNorm32(h) * (1+scale) + shift                  # AdaGN (use_scale_shift_norm=True)
h = zero_init_Conv3x3(Dropout(SiLU(h)))                 # out_layers
return skip_connection(x) + h                            # ch変化時はskipに1x1(or 3x3) conv
```
- `use_scale_shift_norm=True`(デフォルト)のため、タイムステップ埋め込みは加算ではなく
  AdaGN(adaptive group norm、FiLM相当)でresblockに注入される。
- `out_layers`最終convは`zero_module`でゼロ初期化(学習初期は恒等写像に近い挙動)。
- チャンネル数が変わるブロックの skip connection は 1x1 conv (`use_conv=False`なので)。

### AttentionBlock ([unet.py:198](improved_diffusion/unet.py#L198))
- 空間位置全体に対するフルセルフアテンション(ウィンドウなし)。`(b,c,h,w)`を`(b,c,h*w)`にreshapeして適用。
- `qkv = Conv1d(c, 3c, kernel=1)`, `num_heads=4`(デフォルト、`run_train.sh`で上書きなし)。
- **本forkでの変更点**: 標準アテンション(einsum+softmax)ではなく`F.scaled_dot_product_attention`
  (SDPA、torch>=2.0)を使用 ([unet.py:256](improved_diffusion/unet.py#L256))。数値的にはほぼ等価だが、
  Flash-Attention系カーネルにディスパッチされる点が upstream (`openai/improved-diffusion` / `guided-diffusion`) と異なる。
  再現性を厳密に問う場合は要注意([CHANGES_FROM_UPSTREAM.md #1](CHANGES_FROM_UPSTREAM.md)参照)。
- 出力projectionもzero-initialized 1x1 conv。residual加算: `x + attn(x)`。

### timestep embedding ([nn.py:107](improved_diffusion/nn.py#L107) `timestep_embedding`, [unet.py:341](improved_diffusion/unet.py#L341))
- 標準のTransformer型sinusoidal embedding、`dim=model_channels=256`。
- `time_embed = Linear(256,1024) → SiLU → Linear(1024,1024)` (`time_embed_dim = model_channels*4 = 1024`)。
- この1024次元embeddingが全ResBlockにAdaGN経由で注入される。
- `class_cond=False`のため`label_emb`は存在せず、`forward(x, timesteps, y=None)`は`y`を渡すとassertionエラーになる。

### Downsample / Upsample
- `conv_resample=True`(デフォルト)のため、学習可能な畳み込みを使用:
  - Downsample: `Conv3x3(stride=2)`
  - Upsample: `nearest-neighbor x2` → `Conv3x3`
- 平均プーリング/単純補間のみ(`use_conv=False`相当)ではない。

### 正規化・活性化
- 全GroupNormは`GroupNorm32`(`num_groups=32`固定、内部でfloat32にキャストしてから正規化 — fp16学習時の安定化のため)。
- 活性化は全て`SiLU`(`x * sigmoid(x)`)。

---

## 3. パラメータ数・チェックポイント形式

- state_dict中のテンソル数: 486個 (`input_blocks.0`〜`17`, `middle_block.0`〜`2`, `output_blocks.0`〜`17`, `time_embed.0/2`, `out.0/2`。
  実チェックポイント`model130000.pt`で確認済み、`input_blocks.0.0.weight`の形状は`[256,3,3,3]`、`out.2.weight`は`[6,256,3,3]`)。
- **ディスク上のdtype: fp32**。`use_fp16=True`で学習しているが、保存されるのは常にfp32の"master params"
  (`improved_diffusion/fp16_util.py`の`make_master_params`/`_master_params_to_state_dict`、
  [train_util.py:303](improved_diffusion/train_util.py#L303))。モデル内部の順伝播はfp16(torso部分のみ、
  `time_embed`/`out`はfp32のまま)だが、チェックポイントのロード自体は`model.load_state_dict(fp32_dict)`でそのまま可能。
- `dist_util.load_state_dict()`はMPI経由の読み込みラッパー。単一プロセスなら
  `torch.load(path, map_location="cpu")`と等価([CHANGES_FROM_UPSTREAM.md #6](CHANGES_FROM_UPSTREAM.md)参照、
  2GiB超のoptimizer checkpointでmpi4pyのbcastが壊れるバグの回避策あり)。
- チェックポイント3種: `modelNNNNNN.pt`(生の学習済み重み) / `ema_0.9999_NNNNNN.pt`(EMA、生成品質はこちらが基本的に良い) /
  `optNNNNNN.pt`(AdamW optimizer state、生成には不要)。

---

## 4. 拡散過程 (GaussianDiffusion)

これらは`improved_diffusion/gaussian_diffusion.py`の`GaussianDiffusion`(実体は`SpacedDiffusion`サブクラス、
[respace.py:63](improved_diffusion/respace.py#L63))が保持する。CycleDiffusionの実装で`alphas_cumprod`等に直接アクセスする場合はここを参照。

### beta schedule (linear, [gaussian_diffusion.py:27](improved_diffusion/gaussian_diffusion.py#L27))
```python
scale = 1000 / num_diffusion_timesteps   # =1 (steps=1000)
beta_start, beta_end = scale*0.0001, scale*0.02   # = 1e-4, 0.02
betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
alphas_cumprod = np.cumprod(1 - betas)
```
Ho et al. (DDPM論文)と同一のlinear schedule。`steps`を変えても同じ`beta_start/end`域に自動スケールする
実装になっている点に注意(upstreamのDDPM論文の固定1000ステップ前提を任意ステップ数に一般化したもの)。

### モデルの出力パラメータ化
- **`model_mean_type = EPSILON`**(`predict_xstart=False`のデフォルトのため): モデルはノイズ`ε`を予測する
  (`x_0`や`x_{t-1}`ではない)。CycleDiffusionのDDIM系inversion/生成コードはこの`ε`予測を前提に書ける。
- **`model_var_type = LEARNED_RANGE`**(`learn_sigma=True`のため): モデル出力は`(N, 6, H, W)`で、
  channel次元を`ε_pred, v = split(out, 3, dim=1)`のように分割する
  ([gaussian_diffusion.py:262-276](improved_diffusion/gaussian_diffusion.py#L262-L276))。
  `v ∈ ` モデル出力そのまま(だいたい[-1,1]域)を用いて、分散を
  `log_variance = frac*log(betas_t) + (1-frac)*log(posterior_variance_t)`, `frac=(v+1)/2` で
  `FIXED_SMALL`と`FIXED_LARGE`の対数分散を補間する。**CycleDiffusionで決定論的サンプリング(DDIM, η=0)のみ使うなら
  この分散項は使わないため無視してよいが、`model(x,t)`の出力チャンネルは6であり`ε`は前半3chだけという点は必須知識。**
- `predict_xstart_from_eps`: `x0 = sqrt(1/ᾱ_t)*x_t - sqrt(1/ᾱ_t - 1)*ε` ([gaussian_diffusion.py:328](improved_diffusion/gaussian_diffusion.py#L328))。
- `rescale_timesteps=True`(デフォルト): モデルに渡す前に`t`を`t * (1000/num_timesteps)`にスケール
  ([gaussian_diffusion.py:351](improved_diffusion/gaussian_diffusion.py#L351))。学習時`num_timesteps=1000`なので恒等だが、
  `timestep_respacing`でステップを間引いてサンプリングする場合は`_WrappedModel`
  ([respace.py:110](improved_diffusion/respace.py#L110))がoriginal 1000ステップ相当の値に再マップしてから渡す
  (モデルは常に「0〜1000スケールのt」を見る设計)。

### 損失(参考、生成には無関係)
`use_kl=False`, `rescale_learned_sigmas=True` → `loss_type=RESCALED_MSE`
(`mse(ε_pred, ε) + vb_term/1000`, `vb_term`の勾配は`ε`予測経路には流さずvariance経路のみ
[gaussian_diffusion.py:719-732](improved_diffusion/gaussian_diffusion.py#L719-L732))。
`schedule_sampler="uniform"`でtをU(0,999)からサンプル([resample.py:61](improved_diffusion/resample.py#L61))。

### サンプリング
- `p_sample_loop` (祖先サンプリング, DDPM) / `ddim_sample_loop` (DDIM, `eta`で確率性を制御、`eta=0`で決定論的) の両方が実装済み。
- `ddim_reverse_sample` ([gaussian_diffusion.py:524](improved_diffusion/gaussian_diffusion.py#L524)) が決定論的forward ODE
  (画像→ノイズのinversion)に相当し、CycleDiffusionのDDIM inversionを組む際の起点になる。
- `timestep_respacing`: 空文字なら1000ステップフル。`"250"`のように数値文字列で等間隔間引き、
  `"ddim25"`のようにprefixすればDDIM論文のstriding。実装は`space_timesteps()` ([respace.py:7](improved_diffusion/respace.py#L7))。

---

## 5. 入出力の規約

- 画像入力: `(N, 3, 256, 256)`, `float32`, 値域`[-1, 1]`(`pixel/127.5 - 1`、[image_datasets.py:97](improved_diffusion/image_datasets.py#L97))。
- モデル出力: `(N, 6, 256, 256)`。`ε_pred = out[:, :3]`, `v = out[:, 3:6]`(§4参照)。
- `forward(x, timesteps, y=None)`: `class_cond=False`なので`y`は常に`None`(渡すとassert失敗)。
- サンプル→画像変換: `((sample + 1) * 127.5).clamp(0, 255).to(uint8)` ([image_sample.py:56](scripts/image_sample.py#L56))。

---

## 6. このforkでの既知の制約・注意点

- `create_model()`は`image_size ∈ {256, 64, 32}`のみ対応、512pxなどは`ValueError`
  ([script_util.py:105](improved_diffusion/script_util.py#L105))。将来512pxで学習・推論したい場合はここを拡張する必要がある。
- 無条件生成のみ(`class_cond=False`で学習)。[[stage5-classcond-shelved]]の通り、クラス条件付けは
  検討の上見送っているため、条件付きCycleDiffusion(テキスト/クラス条件など)を組むには別途学習が必要。
- Attention実装がupstreamと異なる(SDPA化、§2参照)。ロジックは同値だが、カーネル起因の浮動小数点誤差により
  bit-exactな再現性は保証されない。
- `use_checkpoint=True`(gradient checkpointing)は学習時のみの設定で、重み・推論結果には影響しない。
