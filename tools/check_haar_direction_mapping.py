import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.DWTFreqNet import check_haar_direction_correspondence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check physical H/V orientation of the repository Haar filters"
    )
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--require-aligned-routing",
        action="store_true",
        help="Exit non-zero when the current W8M H/V scan routing is reversed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = check_haar_direction_correspondence(args.size, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_aligned_routing and not result["routing_aligned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
