import json
import os
import cv2
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset structure preprocessing script")
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--gen_video_dir", type=str, required=True, help="Path to generated videos (e.g., Genie_agi_out_sort)")
    parser.add_argument("--output_base", type=str, default="your absolute path", help="Output base directory")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Only process the first N items (0 = all)")
    return parser.parse_args()

def extract_frames(video_path, output_dir):
    """Extract a video into individual frames."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Format as frame_00000.jpg
        frame_name = f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1
    cap.release()

def _extract_ids(gt_path_str):
    """Extract (id1, id2) from gt_path using the parent directory as task group.

    This works regardless of how deeply the gt video is nested, as long as the
    direct parent folder is the task group name and the file stem is the
    episode id, e.g.:

      /.../gt_video/fixed_scene_task/episode1.mp4 -> ("fixed_scene_task", "episode1")
      /data/fixed_scene_task/a/b/c/episode1.mp4   -> ("c",               "episode1")

    Empty parent (e.g. just "episode1.mp4") falls back to "flat".
    """
    p = Path(gt_path_str)
    id2 = p.stem
    id1 = p.parent.name or "flat"
    return id1, id2


def _find_gen_video(gen_video_dir, id1, id2):
    """Locate the generated video, trying multiple naming conventions."""
    candidates = [
        Path(gen_video_dir) / f"{id1}_{id2}.mp4",   # task_name_episodeK.mp4
        Path(gen_video_dir) / f"{id2}.mp4",           # episodeK.mp4
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def process_item(item, gen_video_dir, output_base):
    gt_video_path = Path(item["gt_path"])
    id1, id2 = _extract_ids(item["gt_path"])

    gt_root = Path(output_base) / "gt_dataset" / id1 / id2
    prompt_dir = gt_root / "prompt"
    video_dir_gt = gt_root / "video"

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(video_dir_gt, exist_ok=True)

    prompt_content = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
    with open(prompt_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_content)

    src_image = Path(item["image"])
    if src_image.exists():
        shutil.copy2(src_image, prompt_dir / "init_frame.png")

    if gt_video_path.exists():
        extract_frames(gt_video_path, video_dir_gt)

    gen_root_1 = Path(output_base) / "generated_dataset" / id1 / id2 / "1"
    video_dir_gen_1 = gen_root_1 / "video"
    os.makedirs(video_dir_gen_1, exist_ok=True)

    gen_video = _find_gen_video(gen_video_dir, id1, id2)
    if gen_video:
        extract_frames(gen_video, video_dir_gen_1)
    else:
        print(f"Warning: Generated video not found for {id2} (tried {id1}_{id2}.mp4 and {id2}.mp4)")


def main():
    args = parse_args()
    
    if not os.path.exists(args.summary_json):
        print(f"Error: summary.json not found at {args.summary_json}")
        return

    with open(args.summary_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.limit > 0:
        data = data[:args.limit]

    print(f">>> Starting preprocessing for {len(data)} items...")
    for item in tqdm(data):
        try:
            process_item(item, args.gen_video_dir, args.output_base)
        except Exception as e:
            print(f"Error processing {item.get('gt_path')}: {e}")

    print(f"\n>>> Preprocessing Complete. Structure saved in: {args.output_base}")

if __name__ == "__main__":
    main()