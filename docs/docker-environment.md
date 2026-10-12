# Docker実行環境の構築ガイド

このリポジトリ(`improved-diffusion`)で使っているDocker環境の整え方をまとめたもの。
`guided-diffusion` など、同じ種類のコードベース(PyTorch + MPI前提の拡散モデル学習コード)に
同じ環境を作ることを想定している。移植するコード側の変更は
[porting-to-guided-diffusion.md](porting-to-guided-diffusion.md) を参照。

- 「確認済み」はこのリポジトリのファイルや、このPCでの実行結果で確認した事項。
- 設定の根拠が一般知識によるものは、その旨を書いた。

---

## 1. 全体像

```
ホスト                                    コンテナ (improved-diffusion:<IMAGE_TAG>)
├─ docker-compose.yml  ─ サービス定義      /app            ← ホストのリポジトリ全体をbind mount (./:/app)
├─ Dockerfile          ─ イメージ定義      /app/data       ← ホストのデータセット (.envのHOST_DATA_DIR_*)
├─ .env (git管理外)    ─ PC固有の値        /app/logs/<run> ← 学習ログ・チェックポイント(ホストの ./logs に出る)
├─ run_train.sh        ─ 学習の起動        GPU             ← NVIDIA Container Toolkit経由
├─ run_sample.sh       ─ サンプリング
└─ logs/               ─ 出力(git管理外)
```

設計の考え方は次の4点。

1. **コードはイメージに焼き込まない。** `./:/app` でホストのリポジトリをマウントし、
   イメージには依存パッケージだけを入れる。コードを直しても再ビルドは不要。
2. **PC固有の値は `.env` に出す。** データセットのパス、CUDAのバージョンに合わせたベースイメージなど。
   `.env` はgit管理外で、雛形として `.env.example` をコミットする。
3. **学習パラメータは環境変数で渡す。** `run_train.sh` が環境変数を読み、compose側で値を設定する。
4. **データセットごとにサービスと出力先を分ける。** 自動再開が別データセットの
   チェックポイントを誤って読まないようにする。

---

## 2. 前提(ホスト側)

| 項目 | このPCの値 | 備考 |
|---|---|---|
| Docker Engine | 24.0.2 | 確認済み |
| Docker Compose | v2.18.1(`docker compose`) | 確認済み。`x-` 拡張フィールドとYAMLアンカー(`<<:`)を使う |
| NVIDIAドライバ | 545.23.08(CUDA 12.3まで) | `nvidia-smi` で確認 |
| NVIDIA Container Toolkit | 1.15.0 | 確認済み。`docker info` のRuntimesに `nvidia` がある |

- ホストに **NVIDIAドライバ** と **NVIDIA Container Toolkit** が入っている必要がある。
  CUDA Toolkit自体はホストに不要(イメージ側に入っている)。
- GPUが見えるかの確認(一般的な手順):
  ```bash
  docker run --rm --gpus all pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel nvidia-smi
  ```

---

## 3. Dockerfile

ファイル: [../Dockerfile](../Dockerfile)(確認済み)

```dockerfile
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openmpi-bin libopenmpi-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY setup.py /app/

RUN pip install --no-cache-dir blobfile>=1.0.5 torch tqdm mpi4py Pillow

CMD ["/bin/bash"]
```

要点:

- **ベースイメージは公式の `pytorch/pytorch`**。PyTorch・CUDA・cuDNNが揃っているので、
  自分でCUDAを入れる必要がない。`-devel` なのでコンパイラ(mpi4pyのビルドに必要)も入っている。
- **`ARG BASE_IMAGE` で切り替え可能**にしている(§4)。ホストのドライバに合わせるため。
- **`openmpi-bin` と `libopenmpi-dev`**: `mpi4py` のビルドにヘッダが必要で、複数GPUで学習するときの
  `mpiexec` も提供する。
- **`COPY setup.py` だけ**をコピーしている。コード本体はビルドに含めず、bind mountで渡す。
  コードを変えてもビルドのキャッシュが効く。
- pipで `torch` を指定しているが、ベースイメージに入っているものがそのまま使われる
  (コンテナ内で `2.4.1+cu121` を確認)。`pip install -e .` の依存解決で上書きされないようにする意図。
- **`.dockerignore` はない**(確認済み)。`COPY` が `setup.py` だけなので、実害はない。

### 3.1 ベースイメージの選び方

| ホストのドライバ | 使えるCUDA | ベースイメージ |
|---|---|---|
| >= 550 | 12.4 | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`(デフォルト) |
| >= 530 | 12.1 | `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` |
| >= 520 | 11.8 | `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel` |

`nvidia-smi` 右上の「CUDA Version」以下のイメージを選ぶ。上のバージョンを選ぶと、起動時に次のエラーになる。

```
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.4,
please update your driver to a newer version, or use an earlier cuda container
```

ドライバは新しいCUDAに対して後方互換があるので、迷ったら低いCUDAを選べば安全。
ドライバ更新は他のユーザーのジョブを止める可能性があるため、共有マシンではイメージ側を合わせる方が現実的。

---

## 4. `.env`(PC固有の値)

雛形: [../.env.example](../.env.example)。`.env` にコピーして値を設定する。`.env` は `.gitignore` 済み(確認済み)。

```bash
cp .env.example .env
```

```dotenv
# データセットのホスト側パス(compose側で /app/data にマウントされる)
HOST_DATA_DIR_ANIME=/path/to/your/anime-style-dataset
HOST_DATA_DIR_FFHQ=/path/to/your/real-photo-dataset

# ベースイメージ(ホストのドライバが対応するCUDA以下を選ぶ)
BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
# CUDAバージョン違いのイメージを区別するタグ(別のPCのイメージを上書きしないため)
IMAGE_TAG=cu121
```

- `BASE_IMAGE` を設定しない場合は、デフォルトのCUDA 12.4版になる。
- `IMAGE_TAG` を設定しない場合は `latest` になる。
- データセットのパスはリポジトリに持ち込まない(ローカルのパスがコミットに混ざらない)。
- `HOST_DATA_DIR_*` は、どちらのサービスを使う場合でも**両方設定しておく**のが無難
  (composeはファイル全体の変数を展開するため。未設定時の挙動は未確認)。

---

## 5. docker-compose.yml

ファイル: [../docker-compose.yml](../docker-compose.yml)(確認済み)

### 5.1 サービス構成

| サービス | 用途 | データ | ログ出力先 |
|---|---|---|---|
| `train-anime` | アニメ調データでの学習 | `HOST_DATA_DIR_ANIME` → `/app/data` | `TRAIN_LOGDIR`(サービスごとに別) |
| `train-ffhq` | 実写(FFHQ)データでの学習 | `HOST_DATA_DIR_FFHQ` → `/app/data` | `TRAIN_LOGDIR`(サービスごとに別) |
| `inference` | サンプリング | なし | `SAMPLE_RUN` で指定したrunの下の `samples/` |

3つのサービスは同じイメージ(`improved-diffusion:${IMAGE_TAG}`)を共有する。

### 5.2 ビルドとイメージの共有

```yaml
services:
  train-anime:
    build: &build
      context: .
      args:
        BASE_IMAGE: ${BASE_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}
    image: &image improved-diffusion:${IMAGE_TAG:-latest}
  train-ffhq:
    build: *build
    image: *image
```

- YAMLアンカー(`&build` / `*build`)で、ビルド設定とイメージ名を全サービスで共有する。
  `docker compose build` を1回実行すれば、全サービスが同じイメージを使う。
- `${BASE_IMAGE:-...}` は、`.env` に値があればそれを、なければ `:-` の右の値を使う。

### 5.3 学習パラメータの共通化

```yaml
x-train-params: &train-params
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  IMAGE_SIZE: "256"
  NUM_CHANNELS: "256"
  # ... (モデル・拡散・学習の各パラメータ)
  EXTRA_TRAIN_FLAGS: ""

services:
  train-anime:
    environment:
      <<: *train-params
      TRAIN_LOGDIR: /app/logs/anime-oneres-cosine
```

- 先頭の `x-train-params` に共通値をまとめ、各サービスが `<<: *train-params` で取り込む。
  サービスだけ値を変えたいときは、同じキーを後ろに書いて上書きする。
- 値はすべて**文字列**(`"256"`)にする。数値のまま書くとcomposeが警告を出す場合がある。
- 環境変数の一覧と意味は [porting-to-guided-diffusion.md](porting-to-guided-diffusion.md) §5.1。
- `TRAIN_LOGDIR` は**サービスごとに必ず別の値**にする。自動再開は「出力先の最新チェックポイント」を
  読むため、同じ出力先だと別データセットの重みを読んでしまう。

### 5.4 ボリューム

```yaml
volumes:
  - ./:/app
  - ${HOST_DATA_DIR_ANIME}:/app/data
```

- `./:/app`: リポジトリ全体をマウント。コード修正は即座に反映され、`./logs` に出力が残る。
  イメージの `WORKDIR` も `/app` なので、`run_train.sh` や `scripts/...` を相対パスで呼べる。
- データセットは `/app/data` に固定でマウントし、学習スクリプトは `--data_dir /app/data` だけを見る。
  データを入れ替えても、コード側は変わらない。

### 5.5 GPUの割り当て

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['2']     # 特定のGPUだけ使う場合
          capabilities: [gpu]
        # または
        # - driver: nvidia
        #   count: all
        #   capabilities: [gpu]
```

- `count` と `device_ids` は**どちらか一方**を指定する(両方は不可)。
- 現在の設定: `train-anime` は GPU 2 のみ(`device_ids: ['2']`)、`train-ffhq` は全GPU(`count: all`)。
- コンテナ内からは、割り当てたGPUが `cuda:0` から順に見える。
  シングルプロセスの学習は `cuda:0` を使うので、`device_ids` でGPUを分けるのが並行学習のやり方。
- **注意**: 複数のサービスを同時に動かすとき、両方が `count: all` だと、どちらも `cuda:0` に載ってしまう。
  共有マシンでは、他の人のジョブが動いているGPUを避ける(現在の割り当ては、その運用に合わせたもの)。
- `docker-compose.yml` 先頭の `version: '3.8'` は、新しいComposeでは使われない(警告が出ることがある)。

### 5.6 その他の設定

- `shm_size: '8gb'`: PyTorchのDataLoaderが共有メモリを使うため。現在は `num_workers=0` なので
  実際には使われていないが、ワーカーを増やす場合に必要。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: VRAMの上限ぎりぎりで動かす場合の
  断片化対策(torch >= 2.1)。
- `stdin_open: true` / `tty: true`: `docker compose run` や `attach` で対話的に使うため。
- `container_name`: サービスごとに固定の名前(`improved-diffusion-train-anime` など)。
  `docker compose run` では無視される。
- `command: /bin/bash run_train.sh`: 起動時のコマンド。

---

## 6. 起動スクリプト(`run_train.sh`)

ファイル: [../run_train.sh](../run_train.sh)(確認済み)

起動時に次の順で動く。

1. `pip install -e .`: **毎回**実行する。リポジトリが bind mount されているため、
   イメージのビルド時点のコードと、実行時のコードがずれないようにするため。
2. `OPENAI_LOGDIR` を `TRAIN_LOGDIR` から設定し、ディレクトリを作成する。
3. 環境変数から `MODEL_FLAGS` / `DIFFUSION_FLAGS` / `TRAIN_FLAGS` を組み立てる(未設定ならデフォルト値)。
4. `$OPENAI_LOGDIR` 内の最新の `modelNNNNNN.pt` を探し、あれば `--resume_checkpoint` を付けて再開する。
5. `python scripts/image_train.py --data_dir /app/data ...` を実行する。

`run_sample.sh`(`inference` サービス用)も同じ構成で、`SAMPLE_LOGDIR`、`MODEL_PATH`、
`TIMESTEP_RESPACING`、`NUM_SAMPLES`、`BATCH_SIZE` を環境変数で受け取る。

---

## 7. 基本の使い方

### 7.1 初回セットアップ

```bash
cp .env.example .env        # データのパス、BASE_IMAGE / IMAGE_TAG を編集
docker compose build        # イメージを作成 (または最初のrun/upで自動ビルド)
```

### 7.2 学習

```bash
# フォアグラウンドで実行(端末を閉じると止まる)
docker compose run --rm train-anime

# バックグラウンドで実行
docker compose up -d train-anime
docker compose logs -f train-anime

# 停止
docker compose stop train-anime
```

- 再開は、**同じコマンドをもう一度実行するだけ**。`TRAIN_LOGDIR` の最新チェックポイントから自動で続きを始める。
- 設定の変更は `docker-compose.yml` の `environment` を編集して再実行する。`docker compose run -e KEY=value ...` で
  一時的に上書きすることもできる。
- 2つのサービスを同時に動かすなら、`device_ids` でGPUを分けること(§5.5)。

### 7.3 サンプリング(無条件生成)

設定は `.env`(雛形は `.env.example` の inference の節)で行う。主な項目:

| `.env` のキー | 意味 | デフォルト |
|---|---|---|
| `SAMPLE_RUN` | `./logs/<SAMPLE_RUN>` のチェックポイントを読む | `anime-oneres-cosine` |
| `SAMPLE_GPU` | 使うGPUの番号 | `0` |
| `CKPT_STEP` / `MODEL_PATH` / `EMA_RATE` | 読むチェックポイント。優先順位は `MODEL_PATH` > `CKPT_STEP` > 最新のEMA | 最新の `ema_0.9999_*.pt` |
| `TIMESTEP_RESPACING` / `USE_DDIM` | 生成のステップ間引きとサンプラー(空文字ならフルステップ) | `250` / `False` |
| `NUM_SAMPLES` / `SAMPLE_BATCH_SIZE` | 生成枚数とバッチサイズ | `16` / `16` |
| `SAMPLE_OUTDIR` | 出力先(コンテナ内パス) | `./logs/<SAMPLE_RUN>/samples/<ckpt名>_r<respacing>[_ddim]` |

モデルと拡散過程の設定(`IMAGE_SIZE`、`NOISE_SCHEDULE` など)は、学習サービスと同じ
`docker-compose.yml` の `x-model-params` を共有する。学習時と一致している必要がある。

```bash
docker compose run --rm inference
# 実行時に一時的に上書きする例
docker compose run --rm -e SAMPLE_GPU=1 -e CKPT_STEP=20000 -e NUM_SAMPLES=64 inference
```

出力は `.npz` と、プレビュー用の `.png`(グリッド)。

### 7.4 コンテナ内でのGPU・PyTorchの確認

```bash
docker compose run --rm --no-deps train-ffhq python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

このPCでの結果: `2.4.1+cu121 12.1 True 3`(GPUを全部割り当てた場合。確認済み)。

---

## 8. 他のコードベース(guided-diffusion)に適用するときの手順

1. 次のファイルをコピーする: `Dockerfile`、`docker-compose.yml`、`.env.example`、`run_train.sh`、
   `run_sample.sh`、`.gitignore` の該当部分(`logs/*`、`.env`、`*.egg-info`)。
2. `setup.py` を確認する。`install_requires` と、Dockerfileの `pip install` の内容を合わせる。
   guided-diffusionに追加の依存(`blobfile` 以外)があれば、Dockerfileに足す。
3. `run_train.sh` の `python scripts/image_train.py` と引数名を、guided-diffusionのスクリプトに合わせる
   (`scripts/image_train.py` の引数一覧を確認。モデルの引数が増えている)。
4. `docker-compose.yml` の `x-train-params` から、不要な環境変数を削除し、必要なものを足す。
5. `.env` を作り、`BASE_IMAGE` を**そのPCのドライバに合わせて**設定する。
6. `docker compose build` → §7.4 でGPUとPyTorchを確認 → 短い学習を実行する。
7. 一度止めて、もう一度同じコマンドで再開できるかを確認する(再開テスト。
   コード側の注意は [porting-to-guided-diffusion.md](porting-to-guided-diffusion.md) §4.3, §7.2)。

---

## 9. トラブルシューティング(このプロジェクトで実際に起きたもの)

| 症状 | 原因 | 対処 |
|---|---|---|
| 起動時に `unsatisfied condition: cuda>=12.4` | ホストのドライバが、イメージのCUDAに対応していない | `.env` の `BASE_IMAGE` / `IMAGE_TAG` を低いCUDAに変えて `docker compose build`(§3.1) |
| `docker compose logs -f` が `invalid character '\x00' looking for beginning of value` で止まる | ホスト側のコンテナログファイルが壊れている。学習自体は動いている | `tail -f logs/<TRAIN_LOGDIR名>/log.txt` で見る。または `--tail 50` / `--since 10m` で範囲を絞る |
| 上記のログファイルを消そうとしたがパスが存在しない | ログドライバやデータ保存先が既定と違う | `docker inspect -f '{{.LogPath}}' <コンテナ名>` と `docker info -f '{{.LoggingDriver}} {{.DockerRootDir}}'` で実際の場所を確認する |
| 2つの学習が同じGPUで動いて遅い・メモリ不足 | 両サービスが `count: all` で、どちらも `cuda:0` を使う | サービスごとに `device_ids` で固定(§5.5) |
| 再開しようとしたらチェックポイントのロードに失敗 | 同じ `TRAIN_LOGDIR` で、モデル形状に関わる設定(`NUM_RES_BLOCKS` など)を変えた | `TRAIN_LOGDIR` を新しい名前にする。または設定を元に戻す |

### 9.1 ログの肥大化対策(未適用の提案)

学習は10ステップごとに表を出力するため、長期間動かすとコンテナログが大きくなる。
各サービスに次の設定を足すと、ファイルの上限を設けられる(**まだ `docker-compose.yml` には入れていない**)。
既存のコンテナには反映されず、作り直し(`docker compose up -d`)が必要。

```yaml
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"
```

壊れたログが読めなくなる問題に対して、この設定が有効かどうかは未検証。

### 9.2 ファイルの所有者に注意

コンテナはrootで動くため、`./logs` やその中のディレクトリ、`improved_diffusion.egg-info` は
**root所有**で作られる(このPCで確認済み: `drwxr-xr-x root root`)。ホストの一般ユーザーが
これらを削除・移動するときは `sudo` が必要になる。ホスト側ユーザーで編集したい場合は、
composeに `user: "${UID}:${GID}"` を足す方法があるが、`pip install -e .` がrootを前提にしているため、
この構成では**未検証**。

---

## 10. 既知の制約・未対応

- **複数GPUでの学習は、まだ起動スクリプトが対応していない**。`run_train.sh` は単一プロセスで
  `python` を起動する。`device_ids` に複数のGPUを指定しても、使われるのは1枚(`cuda:0`)。
  詳細と必要な作業は [porting-to-guided-diffusion.md](porting-to-guided-diffusion.md) §8。
- `docker-compose.yml` と `run_train.sh` のデフォルト値は、**サービスの設定を優先する**。
  `run_train.sh` の右辺は、compose側の値を空にしたときにだけ使われる。
- `.dockerignore` が無い(§3)。ビルドコンテキストに `data/` や `logs/` が入ると、ビルドが遅くなる可能性がある
  (`COPY` は `setup.py` だけだが、コンテキスト自体は送信される)。必要なら追加する。
- 起動のたびに `pip install -e .` を実行するため、起動が数秒遅くなる。

---

## 11. 関連ファイル

| ファイル | 内容 |
|---|---|
| [../Dockerfile](../Dockerfile) | イメージ定義(ベースイメージの切り替え、依存パッケージ) |
| [../docker-compose.yml](../docker-compose.yml) | サービス・GPU・ボリューム・学習パラメータ |
| [../.env.example](../.env.example) | `.env` の雛形 |
| [../run_train.sh](../run_train.sh) | 学習の起動スクリプト |
| [../run_sample.sh](../run_sample.sh) | サンプリングの起動スクリプト |
| [porting-to-guided-diffusion.md](porting-to-guided-diffusion.md) | コード側の変更(高速化・バグ修正) |
