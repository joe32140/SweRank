EVAL_MODE="default"
SPLIT="test"
LEVEL="function"

MODEL_NAME=${1:-"Salesforce/SweRankEmbed-Small"}
MODEL_TAG=${2:-"SweRankEmbed-Small"}
BATCH_SIZE=128
DATASET_DIR=${3:-"./datasets/"}
DATASET=${4:-"swe-bench-lite"}
OUTPUT_DIR=${5:-"./results/"}

OUTPUT_FILE=${OUTPUT_DIR}/model=${MODEL_TAG}_dataset=${DATASET}_split=${SPLIT}_level=${LEVEL}_evalmode=${EVAL_MODE}_output.json
RESULTS_FILE=${OUTPUT_DIR}/model=${MODEL_TAG}_dataset=${DATASET}_split=${SPLIT}_level=${LEVEL}_evalmode=${EVAL_MODE}_results.json

echo "Running $MODEL_NAME on TAG: $MODEL_TAG"

python src/eval_beir_sbert_canonical.py \
    --dataset_dir $DATASET_DIR \
    --dataset $DATASET \
    --model $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --output_file $OUTPUT_FILE \
    --results_file $RESULTS_FILE \
    --eval_mode ${EVAL_MODE} --split ${SPLIT} --level ${LEVEL}

echo "Retriever results saved to $RESULTS_FILE"

echo "Running evaluation..."

python src/refactored_eval_localization.py \
        --model $MODEL_TAG \
        --output_dir $OUTPUT_DIR \
        --dataset_dir $DATASET_DIR \
        --output_file $RESULTS_FILE \
        --dataset $DATASET
