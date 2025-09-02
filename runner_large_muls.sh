#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing_new'
LOG_DIR='./logs'
MODEL_DIR='./models'


ALG_NAME="MEMIT"
# MODEL_NAME="gpt2-xl"
MODEL_NAME="gpt-j"
IDENTICAL_NUMS=(2 3 4 5 6 7 8 9 10)
NUM_EDITS_LIST=(1000 1000 1000 1000 1000 1000 1000 1000 1000)
IDENTICAL_RATIOS=(0 1 2 3 4 5 6 7 8 9 10)
MODES=("MEMIT" "TWO-STEP_SUBJECT" "TWO-STEP_RELATION")


for IDX in "${!IDENTICAL_NUMS[@]}"; do
	IDENTICAL_NUM=${IDENTICAL_NUMS[$IDX]}
	NUM_EDITS=${NUM_EDITS_LIST[$IDX]}
	DATA_SIZE=$NUM_EDITS

	LOG_FILE_PATH=$LOG_DIR"/log_large_mul_"$ALG_NAME"_"$MODEL_NAME"_identcial"$IDENTICAL_NUM"_"$DATA_SIZE".txt"
	echo "Removing existing log file : $LOG_FILE_PATH"
	rm -f "$LOG_FILE_PATH"

	for IDENTICAL_RATIO in "${IDENTICAL_RATIOS[@]}"; do
		for MODE in "${MODES[@]}"; do
			echo "alg_name : $ALG_NAME"
			echo "model_name : $MODEL_NAME"
			echo "data_dir : $DATA_DIR"
			echo "identical_num : $IDENTICAL_NUM"
			echo "num_edits : $NUM_EDITS"
			echo "identical_ratio : $IDENTICAL_RATIO"
			echo "mode : $MODE"
			echo "log_file_path : $LOG_FILE_PATH"
			echo "model_dir : $MODEL_DIR"
			echo ""

			python -u -m falcon.tester_large_for_script \
				--alg_name=$ALG_NAME \
				--model_name=$MODEL_NAME \
				--data_dir=$DATA_DIR \
				--identical_num=$IDENTICAL_NUM \
				--num_edits=$NUM_EDITS \
				--identical_ratio=$IDENTICAL_RATIO \
				--mode=$MODE \
				--model_dir=$MODEL_DIR >> $LOG_FILE_PATH 2>&1
		done
	done
done

echo "# Terminate all processes!"

