cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

base_root=${1:-$root_dir/CLIP_logs/scene}

python -u $root_dir/utils/classify_metric.py \
    --base_root $base_root
