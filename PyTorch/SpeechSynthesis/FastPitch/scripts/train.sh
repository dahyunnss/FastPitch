#!/usr/bin/env bash

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=2} #8
: ${BATCH_SIZE:=64} #16
: ${GRAD_ACCUMULATION:=2}
: ${OUTPUT_DIR:="./output"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
: ${DATASET_PATH:=LJSpeech-1.1}
: ${TRAIN_FILELIST:=LJSpeech-1.1/seed_0/train_seed_0.txt}
: ${VAL_FILELIST:=LJSpeech-1.1/seed_0/val_seed_0.txt}
: ${AMP:=true}
: ${SEED:="0"}

: ${LEARNING_RATE:=0.1}

# Adjust these when the amount of data changes
: ${EPOCHS:=1000}
: ${EPOCHS_PER_CHECKPOINT:=20}
: ${WARMUP_STEPS:=1000}
: ${KL_LOSS_WARMUP:=100}

# Train a mixed phoneme/grapheme model
: ${PHONE:=true}
# Enable energy conditioning
: ${ENERGY:=true}
: ${TEXT_CLEANERS:=english_cleaners_v2}
# Add dummy space prefix/suffix is audio is not precisely trimmed
: ${APPEND_SPACES:=false}

: ${LOAD_PITCH_FROM_DISK:=true}
: ${LOAD_MEL_FROM_DISK:=false}

# For multispeaker models, add speaker ID = {0, 1, ...} as the last filelist column
: ${NSPEAKERS:=1}
: ${SAMPLING_RATE:=22050}

# Adjust env variables to maintain the global batch size: NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 256.
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

SEEDS=(0 1 2 3 4 5 6 7 8 9)

for SEED in "${SEEDS[@]}"
do
    # 각 시드에 대한 출력 및 로그 디렉토리 설정
    SEED_OUTPUT_DIR="${OUTPUT_DIR}/seed_${SEED}"
    SEED_LOG_FILE="${SEED_OUTPUT_DIR}/nvlog.json"
    mkdir -p "$SEED_OUTPUT_DIR"

    # 시드별로 학습 및 검증 파일리스트 설정 (필요에 따라 수정)
    TRAIN_FILELIST="${DATASET_PATH}/seed_${SEED}/train_seed_${SEED}.txt"
    VAL_FILELIST="${DATASET_PATH}/seed_${SEED}/val_seed_${SEED}.txt"



  # ARGS=""
  ARGS+=" --cuda"
  ARGS+=" -o $OUTPUT_DIR"
  ARGS+=" --log-file $LOG_FILE"
  ARGS+=" --dataset-path $DATASET_PATH"
  ARGS+=" --training-files $TRAIN_FILELIST"
  ARGS+=" --validation-files $VAL_FILELIST"
  ARGS+=" -bs $BATCH_SIZE"
  ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
  ARGS+=" --optimizer adam"
  ARGS+=" --epochs $EPOCHS"
  ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"

  ARGS+=" --warmup-steps $WARMUP_STEPS"
  ARGS+=" -lr $LEARNING_RATE"
  ARGS+=" --weight-decay 1e-6"
  ARGS+=" --grad-clip-thresh 1000.0"
  ARGS+=" --dur-predictor-loss-scale 0.1"
  ARGS+=" --pitch-predictor-loss-scale 0.1"
  ARGS+=" --trainloader-repeats 100"
  ARGS+=" --validation-freq 10"

  # Autoalign & new features
  ARGS+=" --kl-loss-start-epoch 0"
  ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"
  ARGS+=" --text-cleaners $TEXT_CLEANERS"
  ARGS+=" --n-speakers $NSPEAKERS"

  [ "$AMP" = "true" ]                    && ARGS+=" --amp"
  [ "$PHONE" = "true" ]                  && ARGS+=" --p-arpabet 1.0"
  [ "$ENERGY" = "true" ]                 && ARGS+=" --energy-conditioning"
  [ "$SEED" != "" ]                      && ARGS+=" --seed $SEED"
  [ "$LOAD_MEL_FROM_DISK" = true ]       && ARGS+=" --load-mel-from-disk"
  [ "$LOAD_PITCH_FROM_DISK" = true ]     && ARGS+=" --load-pitch-from-disk"
  [ "$PITCH_ONLINE_DIR" != "" ]          && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch
  [ "$PITCH_ONLINE_METHOD" != "" ]       && ARGS+=" --pitch-online-method $PITCH_ONLINE_METHOD"
  [ "$APPEND_SPACES" = true ]            && ARGS+=" --prepend-space-to-text"
  [ "$APPEND_SPACES" = true ]            && ARGS+=" --append-space-to-text"
  [[ "$ARGS" != *"--checkpoint-path"* ]] && ARGS+=" --resume"

  if [ "$SAMPLING_RATE" == "44100" ]; then
    ARGS+=" --sampling-rate 44100"
    ARGS+=" --filter-length 2048"
    ARGS+=" --hop-length 512"
    ARGS+=" --win-length 2048"
    ARGS+=" --mel-fmin 0.0"
    ARGS+=" --mel-fmax 22050.0"

  elif [ "$SAMPLING_RATE" != "22050" ]; then
    echo "Unknown sampling rate $SAMPLING_RATE"
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"

  : ${DISTRIBUTED:="torchrun --standalone --nnodes=1 --nproc_per_node $NUM_GPUS"}
  echo "Starting training for seed $SEED..."
  $DISTRIBUTED train.py $ARGS "$@"
  echo "Training for seed $SEED completed."
done