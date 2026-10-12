# guided-diffusion への移植ガイド: 高速化・パッケージ更新・安定化の実装まとめ

このリポジトリ(`openai/improved-diffusion` ベース)で行った、**PyTorchのバージョンアップ、高速化、
分散学習まわりの修正、運用面の整備**を、`openai/guided-diffusion` に適用する人向けにまとめたもの。

- upstream: `openai/improved-diffusion` (base commit `1bc7bbb`)
- 変更の素のdiffは [../CHANGES_FROM_UPSTREAM.md](../CHANGES_FROM_UPSTREAM.md) にある。本書はそれを
  「なぜ効くか」「guided-diffusionのどこに当てるか」「どう検証するか」の観点で再構成し、
  パッケージ/環境のバージョン情報と運用設定(`docker-compose.yml` / `run_train.sh`)を加えたもの。
- 「確認済み」と書いたものはこのリポジトリのコード・実測で確認した事項。
  「guided-diffusion側の記述」は **このリポジトリにguided-diffusionのソースが無いため、記憶に基づく**。
  適用前に必ず手元のチェックアウトで該当箇所を確認すること(§6, §7)。

---

## 1. 要点(優先度順)

| 優先 | 項目 | 種別 | 効果 | 該当節 |
|---|---|---|---|---|
| 1 | MPI `bcast` の2GiB制限の回避 | バグ修正(必須) | 再開(resume)機能が動く | §4.3 |
| 2 | 単一プロセス時にDDPでラップしない | 速度 | 実測で1ステップあたり約1桁高速化(RTX 4090 1枚) | §3.2 |
| 3 | Attention を `F.scaled_dot_product_attention` に置換 | 速度・メモリ | 高解像度attention層で有効 | §3.1 |
| 4 | `cudnn.benchmark` / TF32 | 速度 | 入力shape固定の学習で畳み込みを最速化 | §3.3 |
| 5 | 単一プロセス時はgloo、`sync_params` は `p.data` に broadcast | 安定性 | Docker on WSL2でのsegfault・エラー回避 | §4.1, §4.2 |
| 6 | 勾配累積時のlr補正の見直し | 運用 | AdamWでは不要 | §5.4 |

---

## 2. パッケージ・実行環境のバージョン

### 2.1 現在の構成(確認済み)

| 項目 | 値 | 出所 |
|---|---|---|
| ベースイメージ | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`(デフォルト) | [../Dockerfile](../Dockerfile) |
| 現在のGPU機(ドライバ545)で使用 | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` | `.env` の `BASE_IMAGE` |
| PyTorch | 2.4.1 | イメージ同梱。コンテナ内で `2.4.1+cu121` を確認 |
| apt | `openmpi-bin`, `libopenmpi-dev`, `git` | Dockerfile |
| pip | `blobfile>=1.0.5`, `torch`, `tqdm`, `mpi4py`, `Pillow` | Dockerfile |
| `setup.py` の `install_requires` | `blobfile>=1.0.5`, `torch`, `tqdm`(upstreamから変更なし) | [../setup.py](../setup.py) |
| 実行時のインストール | `run_train.sh` が起動ごとに `pip install -e .` | [../run_train.sh](../run_train.sh) |

### 2.2 バージョンを上げる理由と要件

- **torch >= 2.0**: `F.scaled_dot_product_attention` に必要(§3.1)。
- **torch >= 2.1**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` に必要(§3.4)。
- **cuDNN 9**: ベースイメージに同梱。`cudnn.benchmark` を使う(§3.3)。
- **mpi4py**: pipで入れる(Dockerfileのコメントどおり、コードベースが使うため)。OpenMPIのヘッダが必要なので
  `libopenmpi-dev` を先にaptで入れる。

### 2.3 ホストのNVIDIAドライバとCUDAの対応

コンテナ内のCUDAバージョンが、ホストのドライバ対応バージョンを超えると起動できない。実際に次のエラーを踏んだ。

```
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.4, please update your driver ...
```

- 確認方法: `nvidia-smi` 右上の「CUDA Version」以下のイメージを選ぶ。
- 対応表(Dockerfileのコメントと同じ):

  | ドライバ | 使えるCUDA | ベースイメージ |
  |---|---|---|
  | >= 550 | 12.4 | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`(デフォルト) |
  | >= 530 | 12.1 | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` |
  | >= 520 | 11.8 | `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel` |

- 切り替えは [../Dockerfile](../Dockerfile) の `ARG BASE_IMAGE` と、`.env` の `BASE_IMAGE` / `IMAGE_TAG` で行う。
  `IMAGE_TAG` をCUDAごとに分け、別バージョンのイメージを上書きしない。
- ドライバは新しいCUDAに対して後方互換がある。迷ったら低いCUDAのイメージを選べば安全。

### 2.4 GPU世代による違い(一般知識。実機での性能比較は未実施)

- 元の最適化はRTX 4090(Ada)向け。現在のGPU機はQuadro RTX 8000(Turing, sm75)。
- **TF32 は Ampere(sm80)以降の機能**。Turingでは `allow_tf32 = True` を設定しても効果がない(害もない)。
- **bf16 はTuringでは実用にならない**。このリポジトリのfp16(マスターパラメータfp32)の運用はそのまま使える。
- PyTorch 2.4 のFlash-Attention系カーネルはsm80以降が対象で、Turingでは別のカーネル
  (memory-efficient / math)にフォールバックする。SDPA置換の効果の大きさはGPU世代で変わる(§3.1)。

---

## 3. 高速化

### 3.1 Attention を `F.scaled_dot_product_attention` に置換

**対象**: [../improved_diffusion/unet.py](../improved_diffusion/unet.py) の `QKVAttention.forward`(確認済み)

```python
ch = qkv.shape[1] // 3
q, k, v = th.split(qkv, ch, dim=1)
q = q.transpose(-1, -2)
k = k.transpose(-1, -2)
v = v.transpose(-1, -2)
out = F.scaled_dot_product_attention(q, k, v)
return out.transpose(-1, -2)
```

- 元実装はeinsum+softmaxでT×Tの注意行列を毎回実体化する。SDPAは融合カーネルにディスパッチされる。
  `attention_resolutions` に32を含むと、その層は T = 32×32 = 1024 トークンになり、効果が大きい。
- ここでの `qkv` は channel-first で形は `(N*heads, 3*ch, T)`。SDPAはsequence-firstの
  `(batch, T, ch)`を期待するため、`transpose(-1, -2)` で入れ替えて戻す。ヘッドはバッチ次元に畳まれている。
- SDPAの既定スケール `1/sqrt(ch)` は、元コードの `(1/sqrt(sqrt(ch)))` を q と k に掛ける形と等価なので、
  `scale` 引数は渡さない。`import math` は不要になる。
- **数値の一致を実機で確認した**(今回、Quadro RTX 8000 / PyTorch 2.4.1+cu121 で実行):

  入力は `qkv` の形 `(8, 192, 1024)` の乱数、旧実装(einsum+softmax)と新実装(SDPA)の出力差。

  | デバイス | dtype | 最大絶対誤差 |
  |---|---|---|
  | CPU | fp32 | 0 |
  | CUDA (Turing) | fp32 | 0 |
  | CUDA (Turing) | fp16 | 0(524,288要素中、不一致0) |

  別の入力(`flip`)を入れた対照実験では差は約0.41になり、比較自体が機能していることは確認した。
  **これは「この環境のこの形で完全一致した」という結果にとどまる。** どのバックエンドが選ばれたかは
  確認していない。Ampere以降のFlash-Attentionカーネルでは、数値的にbit一致しない小さな差が出るのが普通
  ([../OVERVIEW.md](../OVERVIEW.md) §5 にも「bit-exactではない」と記載)。移植先では §7.1 の検証を実行すること。
- 速度の実測値(SDPA単体の高速化率)はこのリポジトリには記録がない。

**guided-diffusion への適用**(記憶に基づく。要確認):

- guided-diffusion の `unet.py` には `QKVAttentionLegacy` と `QKVAttention` の2系統があり、
  `AttentionBlock` の `use_new_attention_order` で切り替わる。**使う側の両方を置換する**必要がある。
- `QKVAttentionLegacy` は `qkv.reshape(bs * n_heads, ch * 3, length).split(ch, dim=1)` で
  `(bs*n_heads, ch, length)` の q, k, v を作る。この形はこのリポジトリと同じなので、上のコードがそのまま使える。
- `QKVAttention`(新しい順序)は `qkv.chunk(3, dim=1)` から作り、`view(bs * n_heads, ch, length)` に直してから
  einsumしている。**SDPAに渡す前に同じ `(bs*n_heads, ch, length)` に整形**してから、上と同じ
  transpose → SDPA → transpose を行う。
- どちらも出力は `reshape(bs, -1, length)` に戻す部分(元コードにある)を残すこと。

### 3.2 単一プロセスではDDPでラップしない(実測で約1桁高速化)

**対象**: [../improved_diffusion/train_util.py](../improved_diffusion/train_util.py) の `TrainLoop.__init__`(確認済み)

```python
if th.cuda.is_available() and dist.get_world_size() > 1:
    self.use_ddp = True
    self.ddp_model = DDP(...)
else:
    ...
    self.use_ddp = False
    self.ddp_model = self.model
```

- `DDP` は勾配のバケット化・all-reduceのために、モデルのパラメータにautograd hookを直接登録する。
  ラップした後に参照を差し替えても、このhookは外れない。
- 同期する相手(他rank)がいない構成では、hookは純粋なオーバーヘッドになる。
  このリポジトリの計測では、RTX 4090 1枚で1ステップあたり約1桁遅くなっていた。
- `forward_backward()` 内の `no_sync()` の分岐は `last_batch or not self.use_ddp` で守られているので、
  `use_ddp=False` でもそのまま動く。
- guided-diffusion の `TrainLoop` も同じ構造(`if th.cuda.is_available(): self.use_ddp = True ...`)
  のはずなので、同じ条件を足す(要確認)。

### 3.3 `cudnn.benchmark` と TF32

**対象**: [../improved_diffusion/dist_util.py](../improved_diffusion/dist_util.py) の `setup_dist` 冒頭(確認済み)

```python
if th.cuda.is_available():
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
```

- 入力のshape(`image_size`、`batch_size`)が毎ステップ固定なので、`benchmark=True` で最速の畳み込み
  アルゴリズムを最初に一度探索し、結果をキャッシュさせる。入力shapeが変わる用途では逆効果になりうる。
- TF32はAmpere以降で、行列積の精度を少し下げてスループットを稼ぐ(§2.4)。Turingでは効果がない。

### 3.4 CUDAアロケータの断片化対策

**対象**: [../docker-compose.yml](../docker-compose.yml) の `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- torch >= 2.1 が必要。24GBのRTX 4090の上限ぎりぎり(モデル + fp32マスターパラメータ + EMA + optimizer状態
  で活性化用の余裕がほぼない)で、アロケータの断片化による失速を緩和する目的。
- メモリに余裕があるGPU(48GBなど)では効果が小さい可能性がある。

### 3.5 メモリと速度のトレードオフ設定

- `--use_checkpoint True`(勾配チェックポイント): 活性化メモリを節約し、RTX 4090 1枚で
  `num_channels 256` + `attention_resolutions 32,16,8` を収める。再計算の分だけ遅くなる。
  メモリに余裕があるGPUでは `False` にして速度を取れる(§5.1の環境変数 `USE_CHECKPOINT`)。
- `--use_fp16 True`: 順伝播・逆伝播はfp16、マスターパラメータはfp32で持つ構成。
- `microbatch`: VRAMに合わせて調整する。`batch_size = 4 × microbatch` を保てばlr再計算は不要(§5.4)。

### 3.6 未回収のコスト: DataLoaderの `num_workers=0`

- [../improved_diffusion/image_datasets.py](../improved_diffusion/image_datasets.py) で `num_workers` を
  1 → 0 に変更している。Docker on WSL2でのワーカー起動の安定性を優先したもので、原因は未特定。
- 実測ではバッチ取得が1ステップあたり約0.29秒(全体の約7%)で、同期的に消費されている。
- `shm_size: '8gb'` を確保しているので、`num_workers=2` 程度に戻せる余地がある。**意図した高速化ではなく、
  まだ回収できていないコスト**として記録する。WSL2でない環境では最初から戻して問題ない可能性が高い。

---

## 4. 安定性・バグ修正

### 4.1 単一プロセス時はNCCLではなくgloo

**対象**: `dist_util.py` の `setup_dist`(確認済み)

```python
backend = "gloo" if not th.cuda.is_available() or comm.size == 1 else "nccl"
```

- NCCLのP2P/共有メモリのトポロジ探索が、Docker on WSL2でsegfaultすることが知られている。
- 単一プロセスなら同期する相手がおらず、NCCLの利点もない。複数rank(`mpirun`経由)では従来どおりNCCL。

### 4.2 `sync_params` は `p` ではなく `p.data` にbroadcast

**対象**: `dist_util.py` の `sync_params`(確認済み)

```python
dist.broadcast(p.data, 0)   # 元は dist.broadcast(p, 0)
```

- goloバックエンドで、`requires_grad=True` のleafテンソルに直接in-place書き込みすると、
  `no_grad()` の中でも "leaf Variable ... in-place operation" のエラーになる。`.data` なら回避できる。

### 4.3 `load_state_dict`: 単一プロセスではMPI `bcast` を経由しない(必須)

**対象**: `dist_util.py` の `load_state_dict`(確認済み)

```python
comm = MPI.COMM_WORLD
if comm.Get_size() == 1:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)
# 以降は元のrank 0読み込み + bcast
```

- `mpi4py` のpickleベースの `bcast` は、**約2GiBを超えるメッセージで `MPI_ERR_ARG` になる**既知の制限がある。
- 256px・`num_channels=256` のモデルでは、`opt*.pt`(AdamWの1次・2次モーメントを含む)が約3.7GiBになり、
  確実に踏む。本体(約1.9GiB)とEMAは偶然通ってしまう。
- したがって**最初から学習している間は気づけず、`--resume_checkpoint` で再開して初めて失敗する**。
  guided-diffusion でも再開を先にテストしておくこと(§7.2)。
- **複数ranksでは未対策**: `comm.Get_size() > 1` の経路は元のまま(rank 0が読んで`bcast`)なので、
  複数GPUで再開すると同じ制限に当たる可能性が高い(§8)。

---

## 5. 運用面の整備(Docker / compose / run_train.sh)

### 5.1 学習パラメータを環境変数で指定(compose から設定可能)

[../run_train.sh](../run_train.sh) は各引数を `${NAME:-デフォルト}` で読む。未設定・空文字なら、書かれている
デフォルトを使う。[../docker-compose.yml](../docker-compose.yml) の `x-train-params`(YAMLアンカー)に
共通値をまとめ、`train-anime` / `train-ffhq` が `<<: *train-params` で継承する。サービスごとに変えたい値は、
同じキーを `environment` に追記して上書きする。

| 区分 | 環境変数 | 渡される引数 |
|---|---|---|
| モデル | `IMAGE_SIZE` `NUM_CHANNELS` `NUM_RES_BLOCKS` `NUM_HEADS` `ATTENTION_RESOLUTIONS` `DROPOUT` `LEARN_SIGMA` `CLASS_COND` `USE_SCALE_SHIFT_NORM` `USE_CHECKPOINT` | `--image_size` など |
| 拡散 | `DIFFUSION_STEPS` `NOISE_SCHEDULE` | `--diffusion_steps` `--noise_schedule` |
| 学習 | `LR` `BATCH_SIZE` `MICROBATCH` `USE_FP16` `LR_ANNEAL_STEPS` `EMA_RATE` `LOG_INTERVAL` `SAVE_INTERVAL` | `--lr` など |
| 再開 | `RESUME_CHECKPOINT` | `--resume_checkpoint`(空なら自動検出) |
| 追加 | `EXTRA_TRAIN_FLAGS` | 上記以外の引数をそのまま末尾に追加(後ろの値が優先される) |
| 出力先 | `TRAIN_LOGDIR` | `OPENAI_LOGDIR`(ログ・チェックポイントの保存先) |

注意: 既存のrunを再開するとき、モデル形状に関わる値(`IMAGE_SIZE` 〜 `USE_SCALE_SHIFT_NORM`)を変えると
チェックポイントのロードに失敗する。変える場合は `TRAIN_LOGDIR` も別にする。

### 5.2 チェックポイントからの自動再開

- `run_train.sh` が `$OPENAI_LOGDIR` 内の `modelNNNNNN.pt` のうち最大ステップのものを検出し、
  `--resume_checkpoint` に渡す(`sort -V | tail -n 1`)。
- `train_util.py` 側が、ファイル名からステップ数を復元し、同じディレクトリの `opt{step:06d}.pt` と
  `ema_{rate}_{step:06d}.pt` も自動でロードするので、他の指定は不要。
- `RESUME_CHECKPOINT` を指定すると、自動検出より優先して使う。

### 5.3 データセットごとにログ出力先を分ける

- 自動再開は「`$OPENAI_LOGDIR` の最新checkpoint」を拾うため、別データセットのrunと出力先が同じだと、
  他のデータセットのcheckpointを誤って読み込む。`TRAIN_LOGDIR` をサービスごとに別の値にしている。
- データセットのホスト側パスは `.env`(git管理外)の `HOST_DATA_DIR_ANIME` / `HOST_DATA_DIR_FFHQ` で指定する。

### 5.4 勾配累積時のlr補正は、AdamWでは不要

`forward_backward()` は、microbatch単位の平均損失をチャンク数 `K = batch_size / microbatch` で割らずに
足し合わせるため、蓄積される勾配は「batch全体平均の勾配」のK倍になる
([openai/guided-diffusion#111](https://github.com/openai/guided-diffusion/issues/111) と同じ問題)。

- Adamの更新量は `m_hat / (sqrt(v_hat) + eps)` で、勾配をK倍しても `m` と `sqrt(v)` が両方K倍になるため
  **比は変わらない**。AdamW(`weight_decay=0`)では、K倍の影響は実質no-opになる。
- `run_train.sh` は当初 `lr` を `基準lr / K` に補正していたが、AdamWでは不要で、むしろ実効lrを
  下げてしまう。guided-diffusion も既定はAdamWなので、同じ補正をしている場合は見直す価値がある。
- SGD/momentumのようにlrへ勾配が線形に効くオプティマイザでは、補正が必要。
- `weight_decay > 0` の場合、decoupled weight decay項がlrに直接スケールするため、影響を再検討する。
- 現在の `LR` の既定値(`2.5e-5`)がこの補正済みの値か基準値かは、実験計画に合わせて見直すこと。

---

## 6. guided-diffusion への適用手順

### 6.1 ファイルの対応(guided-diffusion側は記憶に基づく。要確認)

| このリポジトリ | guided-diffusion | 変更 |
|---|---|---|
| `improved_diffusion/unet.py` | `guided_diffusion/unet.py` | `QKVAttentionLegacy` / `QKVAttention` をSDPAに(§3.1) |
| `improved_diffusion/train_util.py` | `guided_diffusion/train_util.py` | `world_size > 1` のときだけDDP(§3.2) |
| `improved_diffusion/dist_util.py` | `guided_diffusion/dist_util.py` | `cudnn.benchmark`/TF32(§3.3)、gloo(§4.1)、`p.data`(§4.2)、`load_state_dict`(§4.3) |
| `improved_diffusion/image_datasets.py` | `guided_diffusion/image_datasets.py` | `num_workers`(§3.6。環境次第) |
| `run_train.sh` / `docker-compose.yml` / `Dockerfile` | (新規) | §2, §5 |

### 6.2 guided-diffusionと構造が違う点(記憶に基づく。要確認)

- **fp16まわり**: guided-diffusion は `MixedPrecisionTrainer`(`fp16_util.py`)でfp16を管理する。
  このリポジトリの `TrainLoop` が持つ `_setup_fp16` / `optimize_fp16` / `master_params` とは構造が違う。
  今回の変更はfp16処理そのものには触れていないので、直接の影響はない。ただし §4.3 の
  `opt*.pt` のサイズは、guided-diffusion でマスターパラメータを含む場合に同様に増える。
- **GPUの割り当て**: guided-diffusion の `setup_dist` は、rankごとに環境変数 `CUDA_VISIBLE_DEVICES` を設定し、
  `dev()` は単に `cuda` を返す実装だったはず。このリポジトリの `dev()` は `cuda:{rank % GPUS_PER_NODE}` を返す。
  移植時は、デバイス指定の方式を取り違えないこと。
- **属性/関数名**: `training_losses`、`schedule_sampler` などの呼び出し方は近いが、引数名が違う場合がある。

### 6.3 推奨する順序

1. §4.3(再開が動かないと長期学習で詰む)
2. §3.2(単一GPU運用なら一番効く)
3. §3.1(`attention_resolutions` に32などの高解像度層がある設定ほど効く)
4. §3.3, §4.1, §4.2(環境が該当する場合)
5. §5.4(lr補正を流用している場合のみ)

---

## 7. 検証手順(移植後に必ず行う)

### 7.1 SDPA置換の数値一致

旧実装と新実装を同じ入力で比較する。今回の確認に使ったスクリプトの要点:

```python
import math, torch as th, torch.nn.functional as F

def old(qkv):
    ch = qkv.shape[1] // 3
    q, k, v = th.split(qkv, ch, dim=1)
    scale = 1 / math.sqrt(math.sqrt(ch))
    w = th.einsum("bct,bcs->bts", q * scale, k * scale)
    w = th.softmax(w.float(), dim=-1).type(w.dtype)
    return th.einsum("bts,bcs->bct", w, v)

def new(qkv):
    ch = qkv.shape[1] // 3
    q, k, v = th.split(qkv, ch, dim=1)
    return F.scaled_dot_product_attention(
        q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
    ).transpose(-1, -2)

qkv = th.randn(8, 192, 1024, device="cuda", dtype=th.float16)
print((old(qkv).float() - new(qkv).float()).abs().max())
```

- fp32 と fp16 の両方、実際に使うGPUで実行する。判断基準はプロジェクトで決める
  (この環境では誤差0。Ampere以降のFlash-Attentionでは、fp16の丸め程度の差は想定内)。
- 可能なら、同じシード・同じチェックポイントでサンプリングし、画像を見比べる。

### 7.2 再開テスト

1. 数ステップ学習して `save_interval` でcheckpointを作る(`DIFFUSION_TRAINING_TEST` 環境変数を設定すると
   最初の保存後に終了する。`train_util.py` の `run_loop` 参照)。
2. 同じ出力先で再度起動し、`model` / `ema` / `opt` の3種類すべてが読み込まれることをログで確認する。
3. 「`opt*.pt` が約2GiBを超える設定(大きいモデル)」で必ずやる。小さいモデルでは通ってしまい、
   バグを検出できない。

### 7.3 速度の比較

- 変更前後で `step` あたりの時間を比較する(`progress.csv` / ログの出力間隔)。
- §3.2 のDDP有無は、単一GPUで最も差が出る。変更を1つずつ入れて切り分けること。

---

## 8. 既知の課題・未検証事項

- **複数GPUでの学習は未検証**。コード(MPI + DDP + NCCL、データのrank分割)は対応済みだが、
  `run_train.sh` は単一プロセスで起動するため、現状は複数GPUで動かない。必要な作業:
  - `mpiexec -n <GPU数> python ...` での起動(コンテナ内ではrootなので `--allow-run-as-root` も必要)
  - **複数rankでの再開は、§4.3 の `bcast` 制限に当たる可能性が高い**(各rankが自分でファイルを読む形に直す)
  - `dist_util.GPUS_PER_NODE = 8` は、rankとGPUを「rank % 8」で対応させている。1台のPCでGPU数以下の
    プロセスを起動する限り問題ないが、GPU数を超えるプロセスは起動できない
  - `batch_size` / `microbatch` は**GPU1枚あたりの値**になる。実効バッチは `batch_size × GPU数`で、
    学習率の見直しが必要になる
  - 複数ランクでも、`use_checkpoint` と DDP の組み合わせは未確認
- **並行学習**: 複数のサービス(`train-anime` と `train-ffhq`)を同時に動かすときは、`count: all` ではなく
  `device_ids` でGPUを固定すること。固定しないと、どちらも `cuda:0` に載る。
- **`num_workers`**: §3.6。
- **GPU世代の違い**: 速度の知見はRTX 4090(Ada)での計測。Quadro RTX 8000(Turing)では TF32 とFlash-Attention
  カーネルの効果が変わる(§2.4)。移植先のGPUで再計測すること。
- **SDPAの速度向上率**: このリポジトリには単体の計測値がない(§3.1)。

---

## 9. 関連ファイル

| ファイル | 内容 |
|---|---|
| [../CHANGES_FROM_UPSTREAM.md](../CHANGES_FROM_UPSTREAM.md) | upstreamからの変更点(diff付き) |
| [../OVERVIEW.md](../OVERVIEW.md) | モデル構成・学習方法・データセットの概要 |
| [../run_train.sh](../run_train.sh) | 学習の実行スクリプト(環境変数で設定、自動再開) |
| [../docker-compose.yml](../docker-compose.yml) | サービス定義・学習パラメータの共通値 |
| [../Dockerfile](../Dockerfile) | ベースイメージの切り替え、依存パッケージ |
| [../.env.example](../.env.example) | データパス・`BASE_IMAGE` / `IMAGE_TAG` の例 |
