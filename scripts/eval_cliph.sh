cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=$GPU python -u $root_dir/src/eval_cliph.py \
    --maxActions 150 \
    --eval_save_path $root_dir/CLIP_logs/scene \
    --dataset_path ../DATASET/val_all.json \
    --is_fixed  true\
    --gpu_id $GPU \
    --batchSize 1 \
    --simulator_tool_port 30000

