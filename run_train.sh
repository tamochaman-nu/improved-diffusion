#!/bin/bash
set -e

# docker-composeでカレントディレクトリがマウントされているため、
# コンテナ起動時に毎回パッケージをインストール（またはリンク）します
echo "Installing improved_diffusion package..."
pip install -e .

echo "Starting training..."

# モデルやログの保存先をマウントされているプロジェクトフォルダ内（/app/logs -> ローカルの ./logs）に指定します
export OPENAI_LOGDIR="/app/logs/anime-aligned-curated"
mkdir -p $OPENAI_LOGDIR

# ADM(guided-diffusion)の256px標準構成に寄せたパラメータ設定です。
# resblock_updown等、guided-diffusionで追加されたアーキテクチャ要素はこのコードベース
# (improved-diffusion系のunet.py)には存在しないため反映できません。
# use_checkpoint True で活性化のメモリを節約し、RTX4090 1枚でも
# num_channels 256 + attention_resolutions 32,16,8 が収まるようにしています。
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 2 --attention_resolutions 32,16,8 --learn_sigma True --class_cond False --use_checkpoint True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"

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
TRAIN_FLAGS="--lr 2.5e-5 --batch_size 16 --microbatch 4 --use_fp16 True --lr_anneal_steps 400000"

# 学習スクリプトの実行
python scripts/image_train.py --data_dir /app/data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
