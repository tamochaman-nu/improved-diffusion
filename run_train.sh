#!/bin/bash
set -e

# docker-composeでカレントディレクトリがマウントされているため、
# コンテナ起動時に毎回パッケージをインストール（またはリンク）します
echo "Installing improved_diffusion package..."
pip install -e .

echo "Starting training..."

# モデルやログの保存先をマウントされているプロジェクトフォルダ内（/app/logs -> ローカルの ./logs）に指定します
export OPENAI_LOGDIR="/app/logs/anime"
mkdir -p $OPENAI_LOGDIR

# 先ほどお伝えした画像サイズ(512)などに合わせたパラメータを設定します
# （GPUメモリに応じて batch_size や microbatch は適宜調整してください）
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --microbatch 1 --lr_anneal_steps 400000"

# 学習スクリプトの実行
python scripts/image_train.py --data_dir /app/data $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
