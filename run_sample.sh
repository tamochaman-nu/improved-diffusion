#!/bin/bash
set -e

echo "Installing improved_diffusion package..."
pip install -e .

# --- どの学習run(チェックポイントディレクトリ)からサンプリングするか ---
# デフォルトは現在アクティブなanime-aligned-curated run。他のrun (例: /app/logs/ffhq512)
# から生成したい場合は SAMPLE_LOGDIR で上書きしてください。
SAMPLE_LOGDIR="${SAMPLE_LOGDIR:-/app/logs/anime-aligned-curated}"

# --- どのチェックポイントファイルを読むか ---
# デフォルトは SAMPLE_LOGDIR 内の最大ステップのEMAチェックポイント。
# 特定のファイルを使いたい場合は MODEL_PATH で明示的に指定してください
# (例: MODEL_PATH=/app/logs/anime-aligned-curated/model130000.pt で生EMAでない重みを使う)。
MODEL_PATH="${MODEL_PATH:-$(ls -1 "$SAMPLE_LOGDIR"/ema_0.9999_*.pt 2>/dev/null | sort -V | tail -n 1)}"
if [ -z "$MODEL_PATH" ]; then
    echo "No EMA checkpoint found in $SAMPLE_LOGDIR (set MODEL_PATH explicitly)." >&2
    exit 1
fi
echo "Sampling from: $MODEL_PATH"

# --- モデルアーキテクチャ ---
# チェックポイントファイル自体には形状情報しか無く、image_size等のハイパラは
# 保存されていないため、学習時(run_train.shのMODEL_FLAGS)と必ず一致させる必要があります。
# デフォルトは現行のanime-aligned-curated runの設定。異なるrunのチェックポイントを
# 読む場合はMODEL_FLAGSを丸ごと上書きしてください。
MODEL_FLAGS="${MODEL_FLAGS:---image_size 256 --num_channels 256 --num_res_blocks 2 --attention_resolutions 32,16,8 --learn_sigma True --class_cond False}"
# timestep_respacing: 空文字だとフルの拡散ステップ数(1000)で生成(高品質・低速)。
# "250"のような値でDDPMのリスペーシング、"ddim25"のようにprefixすればDDIMサンプリング。
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --timestep_respacing ${TIMESTEP_RESPACING:-250}"
SAMPLE_FLAGS="--num_samples ${NUM_SAMPLES:-16} --batch_size ${BATCH_SIZE:-16}"

# --- 出力先 ---
# チェックポイントごとにサブディレクトリを分け、サンプル出力が上書き・混在しないようにする。
CKPT_TAG=$(basename "$MODEL_PATH" .pt)
export OPENAI_LOGDIR="${OPENAI_LOGDIR:-$SAMPLE_LOGDIR/samples/$CKPT_TAG}"
mkdir -p "$OPENAI_LOGDIR"

echo "Generating samples..."
python scripts/image_sample.py --model_path "$MODEL_PATH" $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

NPZ=$(ls -1 "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | sort -V | tail -n 1)
if [ -n "$NPZ" ]; then
    echo "Building preview grid..."
    python scripts/npz_to_grid.py "$NPZ"
fi

echo "Done. Output in $OPENAI_LOGDIR"
