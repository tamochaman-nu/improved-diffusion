#!/bin/bash
set -e

# docker-composeでカレントディレクトリがマウントされているため、
# コンテナ起動時に毎回パッケージをインストール（またはリンク）します
echo "Installing improved_diffusion package..."
pip install -e .

echo "Starting training..."

# モデルやログの保存先をマウントされているプロジェクトフォルダ内（/app/logs -> ローカルの ./logs）に指定します。
# train-anime/train-ffhqサービス(docker-compose.yml)がそれぞれ別のTRAIN_LOGDIRを
# 渡すことで、データセットが違うのにログ/チェックポイントディレクトリが同じになり
# （＝再開ロジックが別データセットの最新checkpointを誤って読み込む）事故を防いでいます。
# 単体で直接実行した場合は従来通りanime-aligned-curated-rev1になります。
export OPENAI_LOGDIR="${TRAIN_LOGDIR:-/app/logs/anime-aligned-curated-rev1}"
mkdir -p $OPENAI_LOGDIR

# ADM(guided-diffusion)の256px標準構成に寄せたパラメータ設定です。
# resblock_updown等、guided-diffusionで追加されたアーキテクチャ要素はこのコードベース
# (improved-diffusion系のunet.py)には存在しないため反映できません。
# use_checkpoint True で活性化のメモリを節約し、RTX4090 1枚でも
# num_channels 256 + attention_resolutions 32,16,8 が収まるようにしています。
#
# 以下の各パラメータは環境変数で上書きできます(docker-compose.ymlの各学習サービスの
# environmentで設定)。未設定または空文字の場合は右辺のデフォルト値が使われます。
# 注意: 既存チェックポイントから再開する場合、モデル形状に関わる値(IMAGE_SIZE,
# NUM_CHANNELS, NUM_RES_BLOCKS, ATTENTION_RESOLUTIONS, LEARN_SIGMA, CLASS_COND等)を
# 変えるとロードに失敗します。変更する場合はTRAIN_LOGDIRも別にしてください。
MODEL_FLAGS="--image_size ${IMAGE_SIZE:-256} --num_channels ${NUM_CHANNELS:-256} --num_res_blocks ${NUM_RES_BLOCKS:-2} --num_heads ${NUM_HEADS:-4} --attention_resolutions ${ATTENTION_RESOLUTIONS:-32,16,8} --dropout ${DROPOUT:-0.0} --learn_sigma ${LEARN_SIGMA:-True} --class_cond ${CLASS_COND:-False} --use_scale_shift_norm ${USE_SCALE_SHIFT_NORM:-True} --use_checkpoint ${USE_CHECKPOINT:-True}"
DIFFUSION_FLAGS="--diffusion_steps ${DIFFUSION_STEPS:-1000} --noise_schedule ${NOISE_SCHEDULE:-linear}"

# --- batch_size / microbatch / lr の関係についての注意 ---
# train_util.py の forward_backward() は、勾配累積時に microbatch 単位の平均損失を
# チャンク数 K = batch_size / microbatch で割らずに合計しているため、蓄積される勾配は
# 「batch_size全体での平均勾配」のK倍になります（本家 openai/guided-diffusion の
# 既知の問題: https://github.com/openai/guided-diffusion/issues/111 も参照）。
# そのため、K を固定した上で lr を 基準lr / K に補正しています。
# K=4 は固定のまま、VRAMに合わせて microbatch だけを調整してください
# （変更するたびに、batch_size = 4 * microbatch を保つよう合わせて変更すること。
#  こうすれば lr の再計算は不要です）。
# 例: microbatch 4 → batch_size 16 / microbatch 8 → batch_size 32 / microbatch 16 → batch_size 64
# まずは microbatch=4 から実行し、OOMしなければ2倍ずつ試して限界を確認してください。
TRAIN_FLAGS="--lr ${LR:-2.5e-5} --batch_size ${BATCH_SIZE:-32} --microbatch ${MICROBATCH:-8} --use_fp16 ${USE_FP16:-True} --lr_anneal_steps ${LR_ANNEAL_STEPS:-400000} --ema_rate ${EMA_RATE:-0.9999} --log_interval ${LOG_INTERVAL:-10} --save_interval ${SAVE_INTERVAL:-10000}"

# 上記以外のimage_train.pyの引数(例: "--weight_decay 0.01 --schedule_sampler loss-second-moment")
# はEXTRA_TRAIN_FLAGSでそのまま追加できます。後に書かれた値が優先されるため、
# 上記の引数を上書きする用途にも使えます。
EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"

# --- 前回学習からの再開 ---
# $OPENAI_LOGDIR 内の modelNNNNNN.pt のうち最大ステップのものを検出して
# --resume_checkpoint に渡します。train_util.py 側がファイル名からステップ数を
# 復元し、同じディレクトリの opt{step:06d}.pt / ema_{rate}_{step:06d}.pt も
# 同時に自動ロードするので、これ以外の指定は不要です。
# チェックポイントが無ければ（初回実行時）何も付けず最初から学習します。
# RESUME_CHECKPOINTが指定されていればそれを優先します(特定ステップからやり直す場合など)。
RESUME_FLAGS=""
LATEST_CKPT="${RESUME_CHECKPOINT:-$(ls -1 "$OPENAI_LOGDIR"/model[0-9]*.pt 2>/dev/null | sort -V | tail -n 1)}"
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_FLAGS="--resume_checkpoint $LATEST_CKPT"
else
    echo "No existing checkpoint found, starting fresh training."
fi

# 学習スクリプトの実行
python scripts/image_train.py --data_dir /app/data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $RESUME_FLAGS $EXTRA_TRAIN_FLAGS
