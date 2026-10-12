#!/bin/bash
set -e

echo "Installing improved_diffusion package..."
pip install -e .

# 以下の各設定は環境変数で指定できます(docker-compose.ymlのinferenceサービスのenvironment、
# または.envから)。未設定・空文字の場合は右辺のデフォルト値が使われます。

# --- どの学習run(チェックポイントディレクトリ)からサンプリングするか ---
# docker-compose.ymlでは .env の SAMPLE_RUN (例: anime-oneres-cosine) から
# /app/logs/$SAMPLE_RUN が渡されます。
SAMPLE_LOGDIR="${SAMPLE_LOGDIR:-/app/logs/anime-oneres-cosine}"

# --- どのチェックポイントファイルを読むか(優先順位順) ---
# 1. MODEL_PATH: ファイル(.pt)を直接指定。EMAでない生の重みを使う場合は model<ステップ>.pt を指定する。
#                絶対パス(コンテナ内。例: /app/logs/anime-oneres-cosine/model020000.pt)、または
#                リポジトリルート(/app)からの相対パス(例: logs/anime-oneres-cosine/model020000.pt)。
# 2. CKPT_STEP:  SAMPLE_LOGDIR内の ema_${EMA_RATE}_<ステップ>.pt を指定 (例: 20000 または 020000)
# 3. 未指定:     SAMPLE_LOGDIR内の最大ステップのEMAチェックポイント
# EMA_RATEは学習時の --ema_rate のうち、サンプリングに使うものを1つ指定します。
EMA_RATE="${EMA_RATE:-0.9999}"
if [ -n "$MODEL_PATH" ] && [ "${MODEL_PATH#/}" = "$MODEL_PATH" ]; then
    MODEL_PATH="/app/$MODEL_PATH"
fi
if [ -z "$MODEL_PATH" ]; then
    if [ -n "$CKPT_STEP" ]; then
        MODEL_PATH="$SAMPLE_LOGDIR/ema_${EMA_RATE}_$(printf '%06d' "$((10#$CKPT_STEP))").pt"
    else
        MODEL_PATH="$(ls -1 "$SAMPLE_LOGDIR"/ema_${EMA_RATE}_*.pt 2>/dev/null | sort -V | tail -n 1)"
    fi
fi
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Checkpoint not found: '${MODEL_PATH:-<none>}' (SAMPLE_LOGDIR=$SAMPLE_LOGDIR, EMA_RATE=$EMA_RATE)." >&2
    echo "Set SAMPLE_RUN / CKPT_STEP / MODEL_PATH (see .env.example)." >&2
    exit 1
fi
echo "Sampling from: $MODEL_PATH"

# --- モデルアーキテクチャ・拡散過程 ---
# チェックポイントファイル自体には重みしか無く、image_sizeやnoise_schedule等のハイパラは
# 保存されていないため、学習時(run_train.shのMODEL_FLAGS/DIFFUSION_FLAGS)と必ず一致させる必要が
# あります。特に noise_schedule や learn_sigma が違うと、エラーにならず壊れた画像が出ます。
# docker-compose.ymlでは学習サービスと同じ x-model-params から値が渡されます。
# MODEL_FLAGSを(空でなく)指定した場合は、個別の環境変数よりそちらを丸ごと優先します。
if [ -z "$MODEL_FLAGS" ]; then
    MODEL_FLAGS="--image_size ${IMAGE_SIZE:-256} --num_channels ${NUM_CHANNELS:-256} --num_res_blocks ${NUM_RES_BLOCKS:-1} --num_heads ${NUM_HEADS:-4} --attention_resolutions ${ATTENTION_RESOLUTIONS:-32,16,8} --dropout ${DROPOUT:-0.0} --learn_sigma ${LEARN_SIGMA:-True} --class_cond ${CLASS_COND:-False} --use_scale_shift_norm ${USE_SCALE_SHIFT_NORM:-True}"
fi
DIFFUSION_FLAGS="--diffusion_steps ${DIFFUSION_STEPS:-1000} --noise_schedule ${NOISE_SCHEDULE:-cosine}"

# --- サンプリング設定 ---
# TIMESTEP_RESPACING: 未設定なら250。明示的に空文字にするとフルの拡散ステップ数(DIFFUSION_STEPS)で
#   生成(高品質・低速)。
#   "250"のような値でDDPMのリスペーシング。"ddim25"のようなprefixは「DDIM論文の等間隔ストライド
#   で25ステップに間引く」意味でしかなく、サンプラー自体は変わらない点に注意。
# USE_DDIM: True にすると、サンプラーをDDIM(ddim_sample_loop)に切り替えます。
#   DDIMで生成するなら TIMESTEP_RESPACING="ddim50" のような値と USE_DDIM=True を併用してください。
# CLIP_DENOISED: 各ステップで予測x0を[-1,1]にクリップするか(通常はTrue)。
TIMESTEP_RESPACING="${TIMESTEP_RESPACING-250}"
SAMPLE_FLAGS="--num_samples ${NUM_SAMPLES:-16} --batch_size ${BATCH_SIZE:-16} --use_ddim ${USE_DDIM:-False} --clip_denoised ${CLIP_DENOISED:-True}"

# --- 出力先 ---
# デフォルトは <チェックポイントのあるディレクトリ>/samples/<チェックポイント名>_r<リスペーシング>[_ddim]。
# (MODEL_PATH指定時は、SAMPLE_RUNに関係なくそのファイルのあるrunの下に出力される)
# チェックポイントやサンプラー設定ごとにサブディレクトリを分け、出力が上書き・混在しないようにする。
# SAMPLE_OUTDIR で出力先をそのまま指定することもできます。
CKPT_TAG="$(basename "$MODEL_PATH" .pt)_r${TIMESTEP_RESPACING:-full}"
if [ "$(echo "${USE_DDIM:-False}" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
    CKPT_TAG="${CKPT_TAG}_ddim"
fi
export OPENAI_LOGDIR="${SAMPLE_OUTDIR:-$(dirname "$MODEL_PATH")/samples/$CKPT_TAG}"
mkdir -p "$OPENAI_LOGDIR"

echo "Generating samples..."
python scripts/image_sample.py --model_path "$MODEL_PATH" $MODEL_FLAGS $DIFFUSION_FLAGS \
    --timestep_respacing "$TIMESTEP_RESPACING" $SAMPLE_FLAGS

NPZ=$(ls -1 "$OPENAI_LOGDIR"/samples_*.npz 2>/dev/null | sort -V | tail -n 1)
if [ -n "$NPZ" ]; then
    echo "Building preview grid..."
    python scripts/npz_to_grid.py "$NPZ"
fi

echo "Done. Output in $OPENAI_LOGDIR"
