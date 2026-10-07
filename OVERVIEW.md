# OVERVIEW: improved-diffusionベース 無条件アニメ顔DDPMの実装概要

修士論文での言及を想定した、本プロジェクトで学習しているDDPM(Denoising Diffusion Probabilistic Model)の
アーキテクチャ・学習方法・使用データの網羅的なまとめ。既存実装(`openai/improved-diffusion`)からの
変更点は§5に独立してまとめている。

コードのバージョン管理は本リポジトリ(`~/improved-diffusion`)、データセットの前処理(顔検出・アラインメント・
キュレーション)は独立リポジトリ`~/anime-face-alignment`で行っている(§4参照)。

**作成時点(2026-09-25作成、2026-10-02更新)のスナップショットである点に注意**: 学習は継続中であり、
ステップ数・チェックポイント数は今後も増える。本文中の数値は執筆時点のものとして扱うこと。

---

## 1. プロジェクト概要

- **タスク**: アニメ調の顔画像を対象とした**無条件(unconditional)**画像生成DDPM。
- **ベースコード**: `openai/improved-diffusion`([CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)に記載の
  変更を加えたfork)。
- **実行基盤**: Docker(WSL2上)、単一GPU(NVIDIA GeForce RTX 4090, 24GB)。
- **現在の学習対象**: `logs/anime-aligned-curated-rev1`。256px、無条件、独自キュレーション済みアニメ顔
  データセット(88,081枚、§4)でゼロから学習中(2026-09-24開始、2026-10-02時点でstep 180,420)。
- **先行run**: `logs/anime-aligned-curated`(データキュレーションのv1版、229,372枚、step 136,660まで学習後、
  データのアラインメント手法改善に伴いrev1へ切り替えて停止。§4.4参照)。
- **予備検討として実施したが不採用の経路**: FFHQ(実写顔、70,000枚、512px)での事前学習(`logs/ffhq512`、
  step 120,000まで実施)。最終的な研究対象はアニメ顔生成のため、現在の主軸には含まない(§4.6)。
- **検討したが見送った拡張**: DINOv2埋め込みによるクラスタIDを疑似クラスラベルとしたクラス条件付き学習。
  クラスタが画風ではなく構図/持ち物で分かれる、有効なラベルとしての分散に乏しい等の理由でユーザー判断により
  不採用とし、無条件学習(`--class_cond False`)のまま継続([[stage5-classcond-shelved]])。

---

## 2. モデルアーキテクチャ

### 2.1 全体設計

`improved_diffusion.unet.UNetModel`([improved_diffusion/unet.py](improved_diffusion/unet.py))による、
timestep埋め込み条件付きのU-Net型ノイズ予測ネットワーク。ラベル条件付け機構はコード上に存在するが、
本プロジェクトでは`class_cond=False`のため未使用。

拡散過程は`improved_diffusion.gaussian_diffusion.GaussianDiffusion`
([improved_diffusion/gaussian_diffusion.py](improved_diffusion/gaussian_diffusion.py))が管理し、
Ho et al. (2020) のDDPM定式化に、学習可能な分散(learned variance range, Nichol & Dhariwal 2021)を
追加した構成。

現在の学習で使用しているハイパーパラメータ([run_train.sh](run_train.sh)の`MODEL_FLAGS`/`DIFFUSION_FLAGS`):

```
image_size=256, num_channels=256, num_res_blocks=2, attention_resolutions=32,16,8,
learn_sigma=True, class_cond=False, diffusion_steps=1000, noise_schedule=linear
# 以下は明示指定なし(スクリプト既定値):
num_heads=4, num_heads_upsample=-1(=num_heads), dropout=0.0, use_scale_shift_norm=True,
predict_xstart=False, use_kl=False, rescale_timesteps=True, rescale_learned_sigmas=True,
sigma_small=False, conv_resample=True, timestep_respacing=""(学習時は常にフル1000ステップ)
```

`image_size=256`により`channel_mult=(1,1,2,2,4,4)`が選ばれる
([script_util.py:99-106](improved_diffusion/script_util.py#L99-L106))。**このコードベースの`create_model()`は
`image_size ∈ {256, 64, 32}`のみ対応**しており、512pxなどは`ValueError`になる。

### 2.2 UNetバックボーン構成

`model_channels=256`, `num_res_blocks=2`により、エンコーダ側(`input_blocks`)は6レベル・18ブロック:

| level | mult | channels | 解像度 | ds | attention | 備考 |
|---|---|---|---|---|---|---|
| 0 | 1 | 256 | 256×256 | 1 | – | ResBlock×2 → Downsample |
| 1 | 1 | 256 | 128×128 | 2 | – | ResBlock×2 → Downsample |
| 2 | 2 | 512 | 64×64 | 4 | – | ResBlock×2 → Downsample |
| 3 | 2 | 512 | 32×32 | 8 | ✓ | (ResBlock+Attn)×2 → Downsample |
| 4 | 4 | 1024 | 16×16 | 16 | ✓ | (ResBlock+Attn)×2 → Downsample |
| 5 | 4 | 1024 | 8×8 | 32 | ✓ | (ResBlock+Attn)×2 (Downsampleなし) |

- attentionは`ds ∈ {8,16,32}`(`attention_resolutions="32,16,8"`を`image_size // res`に変換した値)の
  レベルに挿入される([unet.py:375](improved_diffusion/unet.py#L375))。
- **middle_block**(ボトルネック, 8×8, 1024ch): `ResBlock → AttentionBlock → ResBlock`。
- **output_blocks(デコーダ)**: 上記の逆順(level 5→0)、各レベル`num_res_blocks+1=3`ブロック
  (エンコーダ側出力とのchannel-wise concatのため+1)、計18ブロック。`level≠0`の最終ブロックにUpsampleが付き、
  最終出力は256×256に戻る。
- 入力ステム: `Conv2d(3, 256, kernel=3, padding=1)`。
- 出力ヘッド: `GroupNorm32(32,256) → SiLU → zero-initialized Conv2d(256, 6, kernel=3)`
  (`out_channels=6`は`learn_sigma=True`のため、§2.4参照)。

### 2.3 構成要素の詳細

**ResBlock**(`improved_diffusion.unet.ResBlock`、[unet.py:105](improved_diffusion/unet.py#L105)):

```
h = Conv3x3(SiLU(GroupNorm32(x)))                        # in_layers
emb_out = Linear(SiLU(timestep_emb))                      # emb_layers, out=2*out_ch
scale, shift = chunk(emb_out, 2)
h = GroupNorm32(h) * (1 + scale) + shift                  # AdaGN (use_scale_shift_norm=True)
h = zero_init_Conv3x3(Dropout(SiLU(h)))                    # out_layers
return skip_connection(x) + h
```

- `use_scale_shift_norm=True`のため、timestep埋め込みは単純加算ではなくAdaGN(FiLM相当)でResBlockに注入される。
- `out_layers`最終convはゼロ初期化(`zero_module`)、学習初期は恒等写像に近い挙動になる。
- チャンネル数が変化するブロックのskip connectionは1×1 convで整合を取る。

**AttentionBlock**(`improved_diffusion.unet.AttentionBlock`、[unet.py:198](improved_diffusion/unet.py#L198)):

- 空間全域に対するフルセルフアテンション(ウィンドウなし)。`(b,c,h,w)`を`(b,c,h·w)`にreshapeして適用。
- `qkv = Conv1d(c, 3c, kernel=1)`, `num_heads=4`。
- **本forkでの変更点**: 標準実装(手書きeinsum+softmax)ではなく`F.scaled_dot_product_attention`(SDPA、
  torch≥2.0)を使用([unet.py:256](improved_diffusion/unet.py#L256))。数値的にはほぼ等価だが、Flash-Attention系
  カーネルにディスパッチされる点がupstreamと異なる(§5参照)。
- 出力projectionもゼロ初期化1×1 conv。residual加算: `x + attn(x)`。

**timestep embedding**([nn.py:107](improved_diffusion/nn.py#L107)):

- Transformer型sinusoidal embedding、`dim=model_channels=256`。
- `time_embed = Linear(256,1024) → SiLU → Linear(1024,1024)`(`time_embed_dim = model_channels×4 = 1024`)。
- 全ResBlockにAdaGN経由で注入。`class_cond=False`のため`label_emb`(クラス埋め込み)は存在せず、
  `forward(x, timesteps, y=None)`の`y`は常に`None`。

**Downsample / Upsample**: `conv_resample=True`のため学習可能な畳み込みを使用
(Downsample: `Conv3x3(stride=2)`、Upsample: `nearest×2 → Conv3x3`)。単純な平均プーリング/補間のみではない。

**正規化・活性化**: 全GroupNormは`GroupNorm32`(`num_groups=32`固定。fp16学習時の安定化のため内部でfloat32に
キャストしてから正規化)。活性化は全て`SiLU`。

### 2.4 拡散過程のパラメータ化

- **beta schedule (linear)**: Ho et al.と同一の線形スケジュール、`steps`に応じて自動スケール
  ([gaussian_diffusion.py:27](improved_diffusion/gaussian_diffusion.py#L27))。
  ```python
  scale = 1000 / num_diffusion_timesteps    # steps=1000のとき1
  beta_start, beta_end = scale*1e-4, scale*0.02
  betas = linspace(beta_start, beta_end, num_diffusion_timesteps)
  ```
- **`model_mean_type = EPSILON`**(`predict_xstart=False`): モデルはノイズ`ε`を予測する。
- **`model_var_type = LEARNED_RANGE`**(`learn_sigma=True`): モデル出力は`(N,6,H,W)`。
  `ε_pred, v = split(out, 3, dim=1)`のように分割し、`v`(概ね[-1,1])で`FIXED_SMALL`/`FIXED_LARGE`の対数分散を
  補間する: `frac=(v+1)/2`, `log_variance = frac·log(betas_t) + (1-frac)·log(posterior_variance_t)`
  ([gaussian_diffusion.py:262-276](improved_diffusion/gaussian_diffusion.py#L262-L276))。
- **損失関数**: `use_kl=False`, `rescale_learned_sigmas=True` → `RESCALED_MSE`
  (`mse(ε_pred, ε) + vb_term/1000`、`vb_term`の勾配は`ε`予測経路には流さず分散経路のみに流す、
  [gaussian_diffusion.py:719-732](improved_diffusion/gaussian_diffusion.py#L719-L732))。
- **`rescale_timesteps=True`**: モデルに渡す前に`t`を`t·(1000/num_timesteps)`にスケール
  (学習時は`num_timesteps=1000`のため実質恒等)。
- **サンプリング**: `p_sample_loop`(DDPM祖先サンプリング)、`ddim_sample_loop`(DDIM、`eta`で確率性を制御)の
  両方が実装済み。`timestep_respacing`で間引き((例: `"250"`で等間隔、`"ddim25"`でDDIM式ストライド、
  [respace.py](improved_diffusion/respace.py))。

### 2.5 入出力仕様・チェックポイント形式

- 入力: `(N,3,256,256)`, `float32`, 値域`[-1,1]`(`pixel/127.5 - 1`)。
- 出力: `(N,6,256,256)`。前半3chが`ε`予測、後半3chが分散補間パラメータ`v`。
- state_dict中のテンソル数: 486個。ディスク上はfp32(`use_fp16=True`で学習していても、保存されるのは常に
  fp32の"master params"。§3.2参照)。チェックポイント3種: `modelNNNNNN.pt`(生の学習済み重み)、
  `ema_0.9999_NNNNNN.pt`(EMA、生成品質は基本的にこちらが良好)、`optNNNNNN.pt`(AdamW optimizer state)。

より実装寄りの詳細(コード行参照付き)は[DDPM_ARCHITECTURE.md](DDPM_ARCHITECTURE.md)にまとめている。

---

## 3. 学習方法

### 3.1 実行環境

- **ハードウェア**: NVIDIA GeForce RTX 4090(24GB VRAM)単一GPU。
- **実行方式**: Docker Compose(`docker-compose.yml`の`diffusion`サービス)、WSL2上で運用。
- **単一プロセス最適化**: 元コードはMPI(`mpi4py`)+ `torch.distributed`前提の分散学習コードだが、単一GPU運用に
  合わせて以下を適用(詳細は§5・[CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)):
  - DDPでラップしない(`world_size==1`のとき)→実測で1桁高速化
  - `gloo`バックエンド使用(NCCLはWSL2上のDocker環境でsegfaultする既知の問題を回避)
  - `cudnn.benchmark=True`, TF32有効化
- **混合精度**: `use_fp16=True`。モデルの torso 部分(`input_blocks`/`middle_block`/`output_blocks`)をfp16化し、
  `time_embed`/`out`はfp32のまま([unet.py:444](improved_diffusion/unet.py#L444)の`convert_to_fp16`)。
- **勾配チェックポイント**: `use_checkpoint=True`。活性化を保持せず再計算することでメモリを節約し、
  RTX4090 1枚で`num_channels=256`+`attention_resolutions=32,16,8`の256pxモデルを学習可能にしている。

### 3.2 最適化設定

`improved_diffusion.train_util.TrainLoop`([train_util.py](improved_diffusion/train_util.py))が学習ループを
管理する。

- **optimizer**: AdamW(`weight_decay=0.0`、スクリプト既定のまま)。
- **学習率**: `lr=2.5e-5`、`lr_anneal_steps=400000`ステップで線形に0まで減衰
  (`lr = lr_base * (1 - step/lr_anneal_steps)`、[train_util.py:264](improved_diffusion/train_util.py#L264))。
- **EMA(指数移動平均)**: `ema_rate=0.9999`(既定)。毎ステップ`update_ema()`で更新し、生成には基本的に
  EMA重み(`ema_0.9999_*.pt`)を使う。
- **バッチサイズと勾配蓄積**: `batch_size=32`, `microbatch=8`(`K = batch_size/microbatch = 4`)。
  GPUメモリの都合でミニバッチを4分割し、各micro-batchで独立に`loss.backward()`して勾配を蓄積してから
  1回の`opt.step()`を行う([train_util.py:195-233](improved_diffusion/train_util.py#L195-L233))。
  - **既知の注意点**: `forward_backward()`はmicro-batch単位の平均損失をKで割らずに合計するため、
    蓄積される勾配は「batch全体平均のK倍」になる(upstream`guided-diffusion`の既知issue #111と同型の挙動)。
    ただし**AdamWでは`m`/`sqrt(v)`が共にK倍されるため比が変わらず実質no-op**であることを確認済み
    ([CHANGES_FROM_UPSTREAM.md #8](CHANGES_FROM_UPSTREAM.md)参照)。SGD系オプティマイザでは影響するため注意。
- **fp16の損失スケーリング**: 動的スケーリング(`lg_loss_scale`、初期値20、勾配がfiniteなステップごとに
  `+= fp16_scale_growth(1e-3)`、NaN/Inf検出時は`-=1`してそのステップをスキップ)。標準の`torch.cuda.amp.GradScaler`
  とは異なる自前実装([train_util.py:235-249](improved_diffusion/train_util.py#L235-L249))。
- **timestepサンプリング**: `schedule_sampler="uniform"`、各ステップ`t ~ U{0,...,999}`を一様サンプル
  ([resample.py](improved_diffusion/resample.py))。
- **チェックポイント**: `save_interval=10000`ステップごとに`model/ema_0.9999/opt`の3種を保存。
- **再開(resume)**: [run_train.sh](run_train.sh)が`$OPENAI_LOGDIR`内の最大ステップの`modelNNNNNN.pt`を
  自動検出して`--resume_checkpoint`に渡す。`train_util.py`側がファイル名からステップ数を復元し、同ディレクトリの
  `opt*.pt`/`ema_*.pt`も同時に自動ロードする。

### 3.3 ハイパーパラメータ一覧

| 項目 | 値 |
|---|---|
| image_size | 256 |
| num_channels(base) | 256 |
| num_res_blocks | 2 |
| attention_resolutions | 32,16,8 |
| learn_sigma | True |
| class_cond | False |
| diffusion_steps | 1000 |
| noise_schedule | linear |
| optimizer | AdamW (weight_decay=0.0) |
| lr | 2.5e-5 |
| lr_anneal_steps | 400,000(線形減衰) |
| batch_size / microbatch | 32 / 8 (K=4) |
| ema_rate | 0.9999 |
| use_fp16 | True |
| use_checkpoint(勾配チェックポイント) | True |
| schedule_sampler | uniform |
| save_interval | 10,000 step |

(全て[run_train.sh](run_train.sh)の`MODEL_FLAGS`/`DIFFUSION_FLAGS`/`TRAIN_FLAGS`、および
[scripts/image_train.py](scripts/image_train.py)のスクリプト既定値から。)

### 3.4 学習の実施状況(2026-10-02時点)

| run(`OPENAI_LOGDIR`) | データ | 状態 | 到達step |
|---|---|---|---|
| `logs/anime-aligned-curated-rev1`(**現行**) | portraits-aligned-curated-rev1(88,081枚、§4.4) | 学習中(2026-09-24〜) | 180,420 (lr_anneal_steps=400,000に対して約45%) |
| `logs/anime-aligned-curated`(先行run) | portraits-aligned-curated(229,372枚、§4.3) | データのアラインメント改善に伴い停止 | 136,660 |
| `logs/ffhq512`(予備検討、不採用) | FFHQ実写顔(70,000枚、512px→256pxで学習) | 中断・再開失敗のまま放棄 | 120,000 |

(コンテナは2026-09-24から連続稼働中。GPU: RTX4090単体、他の学習ジョブと同時実行はしていない。)

現行run(rev1)は先行runの重みを引き継がず、乱数初期化からの再学習である点に注意
(`model000000.pt`が存在＝ゼロからの学習)。アーキテクチャ・ハイパーパラメータは先行runと同一で、
学習データのみが更新されている。

---

## 4. 使用データ

### 4.1 データソース

生データは`portraits/`ディレクトリ(302,652枚、JPEG、512×512)。

**出典(2026-10-02確認、Web検索により特定)**: 枚数(302,652)・解像度(512px)・格納ディレクトリ名
(`highresolution-anime-face-dataset-512x512`)が完全一致することから、以下の公開データセットであると
確認できた:

> Gwern Branwen, Anonymous, & The Danbooru Community. "Danbooru2019 Portraits: A Large-Scale Anime Head
> Illustration Dataset", 2019-03-12. <https://www.gwern.net/Crops#danbooru2019-portraits>
> (Kaggle配布: <https://www.kaggle.com/datasets/subinium/highresolution-anime-face-dataset-512x512>)

- **収集方法**: Danbooru(イラスト投稿サイト)のSFWサブセットから、`solo`タグ(1人のみ)が付いた
  イラストを対象に、`lbpcascade_animeface`で顔検出し、マージンを広げたクロップ(`y*0.25:y+h,
  x*0.90:x+w*1.25`、顔だけでなく首元/耳/帽子等を含む"ポートレート"風のクロップ)を行ったもの。
- **品質フィルタ**: 学習済みStyleGANのDiscriminatorによるランキングを用いて、異常画像(低品質・誤クロップ等)
  を除去("discriminator ranking")。
- **形式・ライセンス**: JPG、512×512px、**CC0(パブリックドメイン)**。
- 元々はGAN(StyleGAN)によるアニメ顔生成("ThisWaifuDoesNotExist.net")のために構築されたデータセットで、
  本プロジェクトのDDPMと同じ「アニメ顔の無条件生成」という用途に直接合致する。

なお、同じ`/mnt/d/takagi/ffhq_anime/`配下には本家FFHQ(`images1024x1024/`、NVIDIA、CC BY/CC0混在の
Flickr画像由来、§4.6の予備検討で使用)や、Emi 2で生成したAI画像にキャプションを付けたCC0データセット
(`anime-with-caption-cc0/`、15,000枚)、FFHQと1:1で対応付けられた実写/アニメペア画像セット
(`ffhq_anime_rv/`、train/val/test合計70,000枚)も置かれているが、**これらは現在のDDPM学習(`portraits/`
→ キュレーション → アラインメント)の経路には含まれていない**(別目的、おそらく今後のCycleDiffusion系
image-to-image実験用の資材と見られる)。

前処理(顔検出・アラインメント・背景除去・品質キュレーション)は独立リポジトリ`~/anime-face-alignment`
([README.md](file:///home/xr/anime-face-alignment/README.md))で実装している。
[hysts/anime-face-detector](https://github.com/hysts/anime-face-detector)による顔検出、FFHQ式の類似変換に
よるアラインメント、[rembg](https://github.com/danielgatis/rembg)(isnet-animeモデル)による背景除去を使用。

### 4.2 前処理パイプライン全体像

```
portraits/ (302,652, raw)
   │
   ├─→ [旧・簡易アラインメント] → portraits-aligned/ (296,357)  ※現行の学習には不使用
   │
   └─→ [Stage 1: 構図/品質フィルタ]        → 241,378 通過 (79.8%)
        [Stage 2: WD14タグによる様式フィルタ] → 232,831 通過 (96.5%)
        [Stage 3: DINOv2埋め込み+HDBSCANクラスタリング → 人手QCでクラスタ採否判定]
                                              → 230,294 採用 (98.9%, cluster_id=3を除外)
        [Stage 4: マニフェスト作成・シンボリックリンク展開]
                                              → portraits-curation/curated/ (230,294)
             │
             ├─→ [旧アラインメント(黒パディング)を流用+922枚新規処理]
             │     → portraits-aligned-curated/ (229,372) ★先行run(v1)の学習データ
             │
             └─→ [新アラインメント(頭部・髪全体を含むクロップ、フレーム外にはみ出す場合は棄却、背景除去)]
                   → portraits-aligned-curated-rev1/ (88,081 jpg, ~38.2%生存) ★現行runの学習データ
```

### 4.3 Stage 1〜4: 品質・様式キュレーション

`util/curation/`(`~/anime-face-alignment`リポジトリ)配下のスクリプト群。各Stageは中断・再開可能な設計
(CSVへの逐次追記)で、302,652枚全件に対して実行済み。

**Stage 1 — 構図/品質フィルタ**(`stage1_composition_filter.py`):
[hysts/anime-face-detector](https://github.com/hysts/anime-face-detector)で顔検出し、以下の条件で判定:
- 検出顔数(`num_faces`)、顔面積比(`face_area_ratio`)、顔中心オフセット(`face_center_offset`)
- レターボックス(上下左右の帯)の有無。後段の`align_face`(目の間隔ベースのクロップ)が実際に棄却する範囲に
  帯が食い込むかどうかで判定(`letterbox_intrudes`)
- 通過率79.8%(241,378/302,652)。

**Stage 2 — 様式タグ付けフィルタ**(`stage2_wd14_tagger.py`):
WD14タガー(`SmilingWolf/wd-vit-tagger-v3`)でタグ推定し、以下のスコアで除外:
`monochrome`(モノクロ), `greyscale`, `sketch`(線画), `three_d`(3DCG), `chibi`(デフォルメ), `watermark`(透かし),
`signature`(サイン), `speech_bubble`(吹き出し), `english_text`, `japanese_text`(テキスト焼き込み)。
- 通過率96.5%(232,831/241,378、Stage1通過分に対して)。

**Stage 3 — 画風クラスタリング + 人手QC**(`stage3a_extract_embeddings.py` / `stage3b_cluster.py`):
DINOv2(`dinov2_vits14_reg`, 384次元, CLSトークン) で埋め込みを抽出し、PCA(50次元)→UMAP(2次元)→
HDBSCAN(`min_cluster_size=200`)でクラスタリング。232,831枚が8クラスタ+noiseに分かれた。人手QC
([stage3_adopted_clusters.md](file:///mnt/d/takagi/ffhq_anime/highresolution-anime-face-dataset-512x512/portraits-curation/stage3_adopted_clusters.md))
の結果、`cluster_id=3`(2,537枚, 1.1%、日本語テキスト/漫画コマの焼き込みが残存)のみを除外し、他8区分
(cluster 0,1,2,4,5,6,7およびnoise)を採用。DINOv2埋め込みによるクラスタは「画風」よりも「構図・持ち物」
(帽子・ヘッドホン・手/口元の持ち物など)で分かれる傾向が強かったことが判明している(この知見が、
後述のクラス条件付け不採用の一因にもなっている)。
- 採用230,294枚(98.9%、232,831中)。

**Stage 4 — マニフェスト作成**(`stage4_build_manifest.py`):
採用画像を`portraits-curation/curated/`にシンボリックリンクとして展開。

### 4.4 アラインメント: v1(先行run) と rev1(現行run) の違い

Stage 1〜4のキュレーション結果(230,294枚)は共通だが、そこに適用する**顔アラインメント処理**が
先行runと現行runで異なる。

- **旧アラインメント(先行run/v1が使用)**: `portraits-aligned/`(302,652件中296,357件成功、簡易な顔クロップ、
  フレーム外へのはみ出しは黒パディング。実測でコーナーピクセルが`[0,0,0]`(純黒)であることを確認)。
  `portraits-aligned-curated/`(229,372枚)は、キュレーション済み230,294枚のうち229,372枚をこの
  `portraits-aligned/`から再利用し、残る922枚のみ新規処理したもの。
- **新アラインメント(現行run/rev1が使用)**: `~/anime-face-alignment/main.py`による、髪を含む頭部全体を
  FFHQ式の類似変換でクロップし、rembg(isnet-anime)で背景を除去する方式。頭部+髪の全体が元画像の範囲に
  収まらない場合は`--require_full_head`(既定で有効)により**棄却**する(黒パディングでは埋めない)。
  既定パラメータ(`--expand_ratio 2.6`, `--eye_center_y_ratio 0.52`)での実測生存率は約39%
  ([anime-face-alignment/README.md](file:///home/xr/anime-face-alignment/README.md))。
  `portraits-curation/curated/`(230,294枚)にこの新アラインメントを適用した結果が
  `portraits-aligned-curated-rev1/`で、88,081枚(jpgのみ、生存率38.2% ≈ 88,081/230,294)が得られた。
  数値がREADME記載の実測生存率(約39%)とほぼ一致しており、アラインメント方式の変更(髪を含む頭部全体を
  厳密にフレーム内に収める判定)が枚数減少の主因であると判断できる。
- **両者とも出力解像度は256×256**(`main.py --output_size`既定値、現在の学習の`image_size=256`と一致)。

**注記**: 上記のv1/rev1に関する記述は、`~/anime-face-alignment`の`.env`コメント・`README.md`記載の実測値・
実ファイルの画素値検証(黒パディングの有無)から再構成したものであり、変更履歴として明文化されたドキュメントは
存在しない(同リポジトリのgit historyは単一のinitial commitのみ)。おおむね確度は高いが、thesis記載前に
可能であれば実施者本人の記憶で裏取りすることを推奨する。

### 4.5 最終学習データセット(現行run)

- パス: `/mnt/d/takagi/ffhq_anime/highresolution-anime-face-dataset-512x512/portraits-aligned-curated-rev1/`
  (Docker経由で`/app/data`にマウント、`.env`の`HOST_DATA_DIR_ANIME`で指定、`train-anime`サービス)。
- 枚数: 88,081枚(JPEG, 256×256, RGB)。ディレクトリには他に1件の`.zip`ファイルが混在しているが、
  `image_datasets.py`の`_list_image_files_recursively()`は拡張子(`jpg/jpeg/png/gif`)でフィルタするため
  学習には使われない。
- ラベル: なし(無条件学習、`class_cond=False`)。
- 前処理(学習コード側): `ImageDataset`([image_datasets.py](improved_diffusion/image_datasets.py))が
  BOXダウンサンプル→中央クロップで`image_size`(256)に正規化し、`[-1,1]`にスケール
  (今回は既にアラインメント段階で256×256のためリサイズは実質無効)。データ拡張(反転等)は行っていない。

### 4.6 (参考)FFHQ実写データでの予備検討

初期段階で、実写顔データセットFFHQ(70,000枚、512×512を256pxにダウンサンプルして学習)による事前学習を
`logs/ffhq512`で実施していた(step 120,000まで到達)。最終的にアニメ顔生成を研究対象と定めたため、
この実写モデルは現行の学習系列には含まれない(重みの引き継ぎ等も行っていない)。

---

## 5. 既存実装(`openai/improved-diffusion`)からの変更点

コード上の変更は全て[CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md)に詳細(diff付き)でまとめている。
アーキテクチャそのもの(UNetの層構成・拡散過程の定式化)への変更ではなく、**単一GPU・Docker/WSL2環境で
実用的な速度・安定性を得るための実装レベルの変更**である点に注意。要約:

| # | 変更箇所 | 種別 | 内容 |
|---|---|---|---|
| 1 | `unet.py` `QKVAttention.forward` | 速度 | 手書きeinsum+softmaxを`F.scaled_dot_product_attention`(SDPA)に置換。Flash-Attention系カーネルにディスパッチされ、高解像度attention層(T=1024)で特に有効。数値的にはほぼ等価だがbit-exactではない |
| 2 | `train_util.py` `TrainLoop.__init__` | 速度 | 単一プロセス(`world_size==1`)時はDDPでラップしない。同期相手がいない構成でのDDPのautograd hookオーバーヘッドを回避(実測1桁高速化) |
| 3 | `dist_util.py` `setup_dist` | 安定性 | 単一プロセス時はNCCLではなくgloo(Docker on WSL2でNCCLがsegfaultする既知の問題を回避) |
| 4 | `dist_util.py` `sync_params` | バグ修正 | `dist.broadcast(p, 0)` → `dist.broadcast(p.data, 0)`(goloバックエンドでのin-place書き込みエラーを回避) |
| 5 | `dist_util.py` `setup_dist` | 速度 | `cudnn.benchmark=True`、TF32(`allow_tf32`)を有効化 |
| 6 | `dist_util.py` `load_state_dict` | バグ修正(必須) | 単一プロセス時はMPI bcastを経由せず直接読み込み。`mpi4py`のpickle-based bcastは約2GiB超のメッセージで`MPI_ERR_ARG`になる制限があり、256px/num_channels=256クラスのoptimizer checkpoint(~3.7GiB)で確実に踏む。`--resume_checkpoint`で再開して初めて顕在化するため要注意 |
| 7 | `image_datasets.py` `load_data` | 安定性(未最適化) | DataLoaderの`num_workers`を1→0(Docker on WSL2でのワーカー起動まわりの安定性を優先。速度面では未回収のコスト) |
| 8 | `run_train.sh`(運用注記) | 検証結果 | `forward_backward()`の勾配蓄積が`batch_size`平均のK倍になる件(upstream issue #111と同型)を検証した結果、**AdamWでは実質no-op**であると確認(SGD系では要対応) |
| 9 | `Dockerfile`/`docker-compose.yml`/`.env`/`run_train.sh` | 環境整備 | Docker化、`.env`によるデータパスの外出し、起動毎の`pip install -e .`、チェックポイントからの自動再開ロジック等 |

**rev1移行に伴うコード変更はない**(§3.4)。`run_train.sh`の`OPENAI_LOGDIR`(学習データ・チェックポイントの
出力先パス)のみを書き換えて新runを開始しており、モデルアーキテクチャ・学習ハイパーパラメータ・
`improved_diffusion`パッケージ本体のコードは先行run(v1)と同一である。

---

## 6. 関連ドキュメント・ファイル一覧

| ファイル | 内容 |
|---|---|
| [CHANGES_FROM_UPSTREAM.md](CHANGES_FROM_UPSTREAM.md) | upstreamからのコード変更点(diff付き) |
| [DDPM_ARCHITECTURE.md](DDPM_ARCHITECTURE.md) | アーキテクチャのより実装寄りの詳細(コード行参照付き。CycleDiffusion組み込み検討用に作成) |
| [run_train.sh](run_train.sh) | 学習の実行スクリプト(全ハイパーパラメータの一次情報) |
| [run_sample.sh](run_sample.sh) / [scripts/image_sample.py](scripts/image_sample.py) | 無条件生成(推論)スクリプト |
| `~/anime-face-alignment/README.md` | 顔アラインメント・データキュレーションツールの使い方 |
| `~/anime-face-alignment/util/curation/IMPLEMENTATION_PLAN.md` | キュレーションパイプライン(Stage 0〜5)の設計・実装記録。Stage 5(クラス条件付け)不採用の経緯もここに記載 |
| `.../portraits-curation/stage3_adopted_clusters.md` | Stage 3クラスタの採否判定(人手QC記録) |
