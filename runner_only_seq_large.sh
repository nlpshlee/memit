#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing_org'
LOG_DIR='./logs'
MODEL_DIR='./models'


ALG_NAME="MEMIT"
# MODEL_NAME="gpt2-xl"
MODEL_NAME="gpt-j"
IDENTICAL_NUMS=(2 3 4)
NUM_EDITS_LIST=(500 35 5)
MODES=("MEMIT" "TWO-STEP_SUBJECT" "TWO-STEP_RELATION")


for IDX in "${!IDENTICAL_NUMS[@]}"; do
	IDENTICAL_NUM=${IDENTICAL_NUMS[$IDX]}
	NUM_EDITS=${NUM_EDITS_LIST[$IDX]}
	DATA_SIZE=$(($IDENTICAL_NUM * $NUM_EDITS))

	LOG_FILE_PATH=$LOG_DIR"/log_only_seq_large_"$ALG_NAME"_"$MODEL_NAME"_identcial"$IDENTICAL_NUM"_"$DATA_SIZE".txt"
	echo "Removing existing log file : $LOG_FILE_PATH"
	rm -f "$LOG_FILE_PATH"

	for (( BATCH_IDX=1; BATCH_IDX<=IDENTICAL_NUM; BATCH_IDX++ )); do
		for MODE in "${MODES[@]}"; do
			echo "alg_name : $ALG_NAME"
			echo "model_name : $MODEL_NAME"
			echo "data_dir : $DATA_DIR"
			echo "identical_num : $IDENTICAL_NUM"
			echo "num_edits : $NUM_EDITS"
			echo "batch_idx : $BATCH_IDX"
			echo "mode : $MODE"
			echo "log_file_path : $LOG_FILE_PATH"
			echo "model_dir : $MODEL_DIR"
			echo ""

			python -u -m falcon.tester_only_seq_large \
				--alg_name=$ALG_NAME \
				--model_name=$MODEL_NAME \
				--data_dir=$DATA_DIR \
				--identical_num=$IDENTICAL_NUM \
				--num_edits=$NUM_EDITS \
				--batch_idx=$BATCH_IDX \
				--mode=$MODE \
				--model_dir=$MODEL_DIR >> $LOG_FILE_PATH 2>&1
		done
	done
done

echo "# Terminate all processes!"

