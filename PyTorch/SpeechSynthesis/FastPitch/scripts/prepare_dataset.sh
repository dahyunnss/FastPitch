#!/usr/bin/env bash

set -e

: ${DATA_DIR:=LJSpeech-1.1}
: ${ARGS="--extract-mels"}

# 반복 실험을 위한 시드 루프
for SEED in {0..9}; do
    SEED_DIR="LJSpeech-1.1/seed_${SEED}"
    echo "Processing seed $SEED with dataset path: $SEED_DIR"

    python prepare_dataset_dh.py \
        --wav-text-filelists LJSpeech-1.1/ljs_audio_pitch_text_all.txt \
        --n-workers 16 \
        --batch-size 1 \
        --dataset-path $SEED_DIR \
        --extract-pitch \
        --f0-method pyin \
        $ARGS
done