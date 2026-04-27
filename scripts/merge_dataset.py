"""Merge per-scene UAV-ON JSON files into a single JSON file.

Each per-scene file is expected to be a top-level JSON list of episodes.
The merged output is the concatenation of those lists, which is the format
consumed by `src.env_uav.AirVLNENV.load_my_datasets`.
"""
import argparse
import glob
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing per-scene .json files "
             "(e.g. DATASET/UAV-ON-data/valset)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the merged JSON file",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not paths:
        sys.exit(f"No .json files found in {args.input_dir}")

    merged = []
    for p in paths:
        with open(p) as f:
            episodes = json.load(f)
        if not isinstance(episodes, list):
            sys.exit(
                f"{p}: expected a top-level JSON list, "
                f"got {type(episodes).__name__}"
            )
        merged.extend(episodes)
        print(f"  + {os.path.basename(p)}: {len(episodes)} episodes")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f)
    print(
        f"Wrote {len(merged)} episodes from {len(paths)} files "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
