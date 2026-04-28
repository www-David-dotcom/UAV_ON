cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name> [simulator_tool_port]" >&2
    exit 1
fi
dataset_name=$1
port=${2:-30000}

GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU python -u $root_dir/src/eval_cliph.py \
    --maxActions 150 \
    --eval_save_path $root_dir/CLIP_logs/scene \
    --dataset_path ../DATASET/UAV-ON-data/valset/${dataset_name}.json \
    --is_fixed  true\
    --gpu_id $GPU \
    --batchSize 1 \
    --simulator_tool_port $port

