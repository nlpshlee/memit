#!/bin/bash

# 오류가 발생하면 즉시 정지
set -e

DATA_DIR='./data/preprocessing_org'
LOG_DIR='./logs'
MODEL_DIR='./models'

# 이 부분에서 에러 안 나려면 'bash'로 실행해야 함
ARGS_LIST=(
	"MEMIT gpt2-xl 4 5 0 START"
	"MEMIT gpt2-xl 4 5 1 MEMIT"
	"MEMIT gpt2-xl 4 5 2 MEMIT"
	"MEMIT gpt2-xl 4 5 3 MEMIT"
	"MEMIT gpt2-xl 4 5 4 MEMIT"
)

# ARGS_LIST=(
# 	"MEMIT gpt-j 4 5 1 START"
# 	"MEMIT gpt-j 4 5 1 MEMIT"
# 	"MEMIT gpt-j 4 5 1 TWO-STEP_SUBJECT"
# 	"MEMIT gpt-j 4 5 1 TWO-STEP_RELATION"
# )


for ARGS_ in "${ARGS_LIST[@]}"; do
	read -r -a ARGS <<< "$ARGS_"

	ALG_NAME=${ARGS[0]}
	MODEL_NAME=${ARGS[1]}
	IDENTICAL_NUM=${ARGS[2]}
	NUM_EDITS=${ARGS[3]}
	BATCH_IDX=${ARGS[4]}
	MODE=${ARGS[5]}

	DATA_SIZE=$(($IDENTICAL_NUM * $NUM_EDITS))

	# LOG_FILE_PATH=$LOG_DIR"/log_only_seq_large_identcial"$IDENTICAL_NUM"_"$DATA_SIZE"_batch"$BATCH_IDX"_"$MODE".txt"
	LOG_FILE_PATH=$LOG_DIR"/log_only_seq_large_"$ALG_NAME"_"$MODEL_NAME"_identcial"$IDENTICAL_NUM"_"$DATA_SIZE".txt"

	if [[ "$MODE" == "START" ]]; then
		echo "START mode: removing existing log file"
		rm -f "$LOG_FILE_PATH"
		continue
	fi

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

echo "# Terminate all processes!"

