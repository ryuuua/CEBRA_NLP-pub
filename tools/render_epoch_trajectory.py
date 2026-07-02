from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cebra_nlp_public.visualization.trajectory.epoch import (  # noqa: E402
    render_saved_epoch_trajectory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-render a saved epoch trajectory directory into GIF/MP4 without retraining."
    )
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--one-epoch-one-frame",
        dest="one_epoch_one_frame",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--allow-frame-downsampling",
        dest="one_epoch_one_frame",
        action="store_false",
    )
    parser.add_argument("--connect-segments", dest="connect_segments", action="store_true")
    parser.add_argument("--no-connect-segments", dest="connect_segments", action="store_false")
    parser.set_defaults(connect_segments=None)
    parser.add_argument("--rotate-camera", dest="rotate_camera", action="store_true")
    parser.add_argument("--fixed-camera", dest="rotate_camera", action="store_false")
    parser.set_defaults(rotate_camera=None)
    parser.add_argument("--camera-elev", type=float)
    parser.add_argument("--camera-azim", type=float)
    parser.add_argument("--trail-length", type=int)
    parser.add_argument("--axis-padding", type=float)
    parser.add_argument("--frame-width", type=int)
    parser.add_argument("--frame-height", type=int)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--mp4-crf", type=int)
    parser.add_argument("--mp4-preset", type=str)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = render_saved_epoch_trajectory(
        args.trajectory_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        one_epoch_one_frame=args.one_epoch_one_frame,
        connect_segments=args.connect_segments,
        rotate_camera=args.rotate_camera,
        camera_elev=args.camera_elev,
        camera_azim=args.camera_azim,
        trail_length=args.trail_length,
        axis_padding=args.axis_padding,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        dpi=args.dpi,
        mp4_crf=args.mp4_crf,
        mp4_preset=args.mp4_preset,
    )
    for kind, relative_path in outputs.items():
        print(f"{kind}: {args.trajectory_dir / relative_path}")


if __name__ == "__main__":
    main()
