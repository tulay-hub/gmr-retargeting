import argparse
import json
import pathlib
import pickle

from path_utils import repository_root


DEFAULT_DATASET_DIR = repository_root() / "projects" / "03_walk" / "data" / "processed" / "lafan_walk_lens110_21dof"


DEFAULT_CLIPS = [
    ("walk1_subject1", 91, 3621),
    ("walk1_subject2", 1450, 7830),
    ("walk1_subject5", 83, 600),
    ("walk2_subject1", 83, 1304),
]


def trim_pkl(input_path, output_path, start_frame, end_frame):
    with input_path.open("rb") as file:
        data = pickle.load(file)

    stop_frame = end_frame + 1
    trimmed = dict(data)
    for key in ("root_pos", "root_rot", "dof_pos"):
        trimmed[key] = data[key][start_frame:stop_frame].copy()
    trimmed["trim_source_file"] = str(input_path)
    trimmed["trim_start_frame"] = int(start_frame)
    trimmed["trim_end_frame"] = int(end_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(trimmed, file)
    return len(trimmed["root_pos"])


def trim_motion_txt(input_path, output_path, start_frame, end_frame):
    data = json.loads(input_path.read_text())
    stop_frame = end_frame + 1
    trimmed = dict(data)
    trimmed["Frames"] = data["Frames"][start_frame:stop_frame]
    trimmed["TrimSourceFile"] = str(input_path)
    trimmed["TrimStartFrame"] = int(start_frame)
    trimmed["TrimEndFrame"] = int(end_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trimmed, indent=2))
    return len(trimmed["Frames"])


def parse_clip(clip):
    parts = clip.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("clip must be name:start:end, for example walk1_subject1:91:3621")
    name, start_frame, end_frame = parts
    return name, int(start_frame), int(end_frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--clip",
        action="append",
        type=parse_clip,
        default=None,
        help="Clip spec: name:start:end. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    clips = args.clip or DEFAULT_CLIPS
    for name, start_frame, end_frame in clips:
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"Invalid clip range for {name}: {start_frame}..{end_frame}")

        output_name = f"{name}_f{start_frame}_to_f{end_frame}"
        pkl_input = args.dataset_dir / "pkl" / f"{name}.pkl"
        txt_input = args.dataset_dir / "txt" / f"{name}.txt"
        amp_input = args.dataset_dir / "motion_amp_expert_split" / f"{name}.txt"
        pkl_output = args.dataset_dir / "pkl" / f"{output_name}.pkl"
        txt_output = args.dataset_dir / "txt" / f"{output_name}.txt"
        amp_output = args.dataset_dir / "motion_amp_expert_split" / f"{output_name}.txt"

        for path in (pkl_input, txt_input):
            if not path.exists():
                raise FileNotFoundError(path)
        amp_available = amp_input.exists()
        if not args.overwrite:
            output_paths = [pkl_output, txt_output]
            if amp_available:
                output_paths.append(amp_output)
            for path in output_paths:
                if path.exists():
                    raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

        pkl_frames = trim_pkl(pkl_input, pkl_output, start_frame, end_frame)
        txt_frames = trim_motion_txt(txt_input, txt_output, start_frame, end_frame)
        summary = f"{name}: {start_frame}->{end_frame} -> {output_name} (pkl={pkl_frames}, txt={txt_frames}"
        if amp_available:
            amp_frames = trim_motion_txt(amp_input, amp_output, start_frame, end_frame)
            summary += f", amp={amp_frames}"
        summary += ")"
        print(summary)


if __name__ == "__main__":
    main()
