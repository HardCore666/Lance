# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
from pathlib import Path


def build_video_index(root: Path, subdirs: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for subdir in subdirs:
        media_dir = root / subdir
        if not media_dir.is_dir():
            continue
        for name in os.listdir(media_dir):
            if name not in index:
                index[name] = str(Path(subdir) / name)
    return index


def resolve_video(root: Path, video_name: str, subdirs: list[str], video_index: dict[str, str] | None) -> str | None:
    if video_index is not None:
        return video_index.get(video_name)
    for subdir in subdirs:
        rel = Path(subdir) / video_name
        if (root / rel).exists():
            return str(rel)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenVid CSV annotations to Lance finetune JSONL.")
    parser.add_argument("--openvid_root", required=True, help="OpenVidData directory.")
    parser.add_argument("--csv", default="data/train/OpenVid-1M.csv", help="CSV path relative to openvid_root or absolute.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max_samples", type=int, default=0, help="Stop after this many valid samples; 0 means all.")
    parser.add_argument("--min_seconds", type=float, default=0.0)
    parser.add_argument("--max_seconds", type=float, default=0.0, help="0 means no upper bound.")
    parser.add_argument("--media_subdirs", nargs="+", default=["video", "remaining_videos"])
    parser.add_argument(
        "--index_mode",
        choices=["auto", "full", "stream"],
        default="auto",
        help="full scans media directories once; stream checks files row by row. auto uses stream for max_samples and full for all.",
    )
    parser.add_argument("--missing_output", default="", help="Optional JSON path for missing video filenames.")
    parser.add_argument(
        "--trust_subdir",
        default="",
        help="Trust all CSV video filenames to live under this media subdir and skip existence checks.",
    )
    args = parser.parse_args()

    root = Path(args.openvid_root).expanduser().resolve()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    index_mode = args.index_mode
    if index_mode == "auto":
        index_mode = "stream" if args.max_samples else "full"
    video_index = build_video_index(root, args.media_subdirs) if index_mode == "full" else None
    total, written, skipped_missing, skipped_duration = 0, 0, 0, 0
    missing: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f_in, output.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            total += 1
            video_name = (row.get("video") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not video_name or not caption:
                skipped_missing += 1
                continue

            rel_video = str(Path(args.trust_subdir) / video_name) if args.trust_subdir else resolve_video(root, video_name, args.media_subdirs, video_index)
            if rel_video is None:
                skipped_missing += 1
                if len(missing) < 10000:
                    missing.append(video_name)
                continue

            seconds = float(row.get("seconds") or 0.0)
            if args.min_seconds and seconds < args.min_seconds:
                skipped_duration += 1
                continue
            if args.max_seconds and seconds > args.max_seconds:
                skipped_duration += 1
                continue

            record = {
                "prompt": caption,
                "video": rel_video,
            }
            for key in ("aesthetic score", "motion score", "temporal consistency score", "camera motion", "frame", "fps", "seconds"):
                if key in row:
                    record[key] = row[key]
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    if args.missing_output:
        missing_path = Path(args.missing_output).expanduser().resolve()
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with missing_path.open("w", encoding="utf-8") as f:
            json.dump(missing, f, ensure_ascii=False, indent=2)

    print(
        f"converted={written} total_seen={total} "
        f"skipped_missing={skipped_missing} skipped_duration={skipped_duration} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
