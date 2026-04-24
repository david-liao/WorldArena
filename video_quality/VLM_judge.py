import argparse
import json
import os
import shutil
import time
from pathlib import Path
from datetime import datetime
import cv2
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_PATH = "your absolute path"


def load_instruction_json(json_path):
    """Parse summary.json.

    Returns:
        instruction_map: {generated_filename: instruction_text}
        valid_video_basenames: list of generated filenames in the order they
            appear in the summary JSON (duplicates removed, first occurrence
            wins). Returning an ordered list (instead of a set) lets callers
            iterate in summary order, which matches the ordering used by
            preprocess_datasets.py so that LIMIT-based debug runs across
            pipelines always select the same videos.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instruction_map = {}
    valid_video_basenames = []

    for item in data:
        gt_path = item.get("gt_path", "")
        prompt = item.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            instruction = prompt[0]
        elif isinstance(prompt, str):
            instruction = prompt
        else:
            instruction = ""

        path_obj = Path(gt_path)
        if len(path_obj.parts) >= 5:
            generated_filename = f"{path_obj.parts[-1].split('.')[0]}.mp4"
        else:
            base_name = os.path.basename(gt_path)
            generated_filename = f"unknown_{base_name}"

        if generated_filename not in instruction_map:
            valid_video_basenames.append(generated_filename)
        instruction_map[generated_filename] = instruction

    return instruction_map, valid_video_basenames


def sample_frames(video_path, num_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    if total_frames <= num_frames:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames[:num_frames]


def build_multi_metric_prompt(instruction_text=None):
    prompt = """ """
    if instruction_text:
        prompt += f"\n**VIDEO INSTRUCTION:** {instruction_text}\n"
    prompt += """
You are an expert evaluator for robot interaction videos. You are evaluating videos generated for **embodied AI manipulation scenarios**, specifically focusing on robotic arms interacting with objects in tabletop environments.
**EVALUATION CONTEXT:**
- Target scenario: Robotic manipulation (e.g., pick-place, push, grasp)
- Expected agent: **Robotic arm/end-effector**, NOT human hands
- Expected environment: Tabletop with objects, typical for robot manipulation tasks
- Expected physics: Realistic robot-object interactions following physical laws
**CRITICAL EVALUATION PRINCIPLES:**
1. Base ALL judgments ONLY on what is visually observable in the sampled frames
2. DO NOT infer information not shown (no assumptions about unseen parts)
3. Evaluate temporal coherence across the sampled frames
4. For instruction following: Compare STRICTLY against the provided text instruction
**EVALUATION DIMENSIONS & SCORING RUBRICS:**
1. Interaction_Quality (Quality of robot-object interactions)
- Score 1: Objects pass through robot or other objects; no proper contact
- Score 2: Contact exists but interaction is unrealistic (e.g., sliding without friction, incorrect force response)
- Score 3: Mostly plausible interactions with minor issues (e.g., slight penetration, imperfect grasping)
- Score 4: Realistic contact physics (proper friction, force transfer, object deformation)
- Score 5: Perfect interaction physics; indistinguishable from real robot manipulation
2. PERSPECTIVITY (3D consistency and camera geometry)
- Score 1: Scene has no coherent 3D structure; objects float inconsistently
- Score 2: 3D structure is unstable (e.g., scale changes, incorrect occlusion)
- Score 3: Reasonable 3D consistency with minor issues (e.g., slight perspective drift)
- Score 4: Stable camera perspective with consistent depth relationships
- Score 5: Perfect camera geometry and 3D consistency
3. INSTRUCTION FOLLOWING (Adherence to given instruction:**VIDEO INSTRUCTION**)
- **HALLUCINATION CHECK**: If the video shows human hands instead of robotic arms, score <= 2 immediately
- Score 1: Completely different from instruction (wrong action, wrong objects, wrong scene)
- Score 2: Partially related but major errors (e.g., wrong target object, incorrect manipulation type)
- Score 3: Follows general intent but with execution errors (e.g., correct action sequence but imprecise)
- Score 4: Mostly correct with minor deviations (e.g., slight position error, extra unnecessary motion)
- Score 5: Perfect execution of all specified elements (action, object, scene, outcome)
**SPECIFIC ROBOT-RELATED CHECKS:**
- Robotic arm should have mechanical appearance, NOT human limbs
- End-effector (gripper) should maintain consistent form throughout interaction
- Robot motion should show appropriate joint movement and kinematics
- Object manipulation should respect object mass and inertia
- Contact should be maintained appropriately during grasping/lifting
**OUTPUT FORMAT REQUIREMENTS:**
You MUST output a SINGLE, VALID JSON object with EXACTLY three keys:
- 'Interaction_Quality'
- 'Perspectivity'
- 'Instruction_Following'
Each value must be an object with exactly two keys:
- '"score"': integer 1-5
- '"reason"': concise explanation citing SPECIFIC visual evidence from frames
**EXAMPLE OUTPUT:**
{
"Interaction_Quality": {
"score": 2,
"reason": "Object slides without friction during pushing;
gripper penetrates object slightly"
},
"Perspectivity": {
"score": 4,
"reason": "Stable camera perspective with consistent depth ordering"
},
"Instruction_Following": {
"score": 1,
"reason": "Video shows human hand instead of robotic arm (hallucination)"
}
}
**CRITICAL INSTRUCTIONS:**
1. Output ONLY the JSON object, no other text
2. Base scoring on observed visual evidence only
3. For instruction following: Strictly compare with the provided instruction
4. Consider temporal coherence across all sampled frames
5. Penalize hallucinations (e.g., human hands instead of robot) heavily
Now evaluate the provided video frames based on the above criteria.
"""

    return prompt


@torch.inference_mode()
def run_qwen_vl(model, processor, prompt, images):
    messages = [{
        "role": "user",
        "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": prompt}],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if "pixel_values" not in inputs:
        image_inputs = processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = image_inputs.pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_trim = generated_ids[:, input_len:]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def safe_parse_scores(raw_text):
    try:
        obj = json.loads(raw_text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def normalize_metrics(parsed):
    metrics = {}
    for key in ["Interaction_Quality", "Perspectivity", "Instruction_Following"]:
        val = parsed.get(key, {}) if isinstance(parsed, dict) else {}
        score = val.get("score", 0)
        reason = val.get("reason", "")
        metrics[key] = {
            "score": score,
            "reason": reason,
            "score_normalized": round(score / 5.0, 4) if isinstance(score, (int, float)) else 0.0,
        }
    return metrics


def vlm_judge(model_name, video_dir, summary_json, output_root, tmp_root, metrics_filter, num_frames=16, model_path=None, max_videos=0, shard_id=0, num_shards=1):
    os.makedirs(output_root, exist_ok=True)
    tmp_suffix = f"_shard{shard_id}" if num_shards > 1 else ""
    tmp_dir = os.path.join(tmp_root, f"{model_name}{tmp_suffix}")
    os.makedirs(tmp_dir, exist_ok=True)

    model_path = model_path or DEFAULT_MODEL_PATH
    print(f"Loading model from {model_path} ...")
    t_load = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model loaded in {time.time() - t_load:.1f}s (attn=sdpa)")

    instruction_map, valid_names = load_instruction_json(summary_json)

    # Iterate in summary-JSON order (not lexicographic listdir order) so that
    # LIMIT/max_videos selects the same videos as preprocess_datasets.py and
    # preprocess_datasets_diversity.py, keeping aggregate_results.py's joins
    # consistent. Skip entries whose .mp4 isn't present under video_dir.
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    existing_files = {
        f for f in os.listdir(video_dir) if f.lower().endswith(video_exts)
    }
    missing = [n for n in valid_names if n not in existing_files]
    if missing:
        print(f"[WARN] {len(missing)} video(s) listed in summary are missing under {video_dir}; first 3: {missing[:3]}")
    videos = [os.path.join(video_dir, n) for n in valid_names if n in existing_files]
    if max_videos > 0:
        videos = videos[:max_videos]
    if num_shards > 1:
        videos = [v for i, v in enumerate(videos) if i % num_shards == shard_id]

    total_videos = len(videos)
    shard_tag = f" shard {shard_id}/{num_shards}" if num_shards > 1 else ""
    print(f"Evaluating {total_videos} videos (num_frames={num_frames}){shard_tag}")
    results = []
    t_total_start = time.time()
    video_times = []

    pbar = tqdm(videos, desc=f"{model_name} evaluating", ncols=120)
    for video_path in pbar:
        t_video = time.time()
        item = {"video": os.path.basename(video_path), "metrics": {}, "raw_response_file": None, "error": None}
        frames = sample_frames(video_path, num_frames=num_frames)
        if not frames:
            item["error"] = "no frames"
            results.append(item)
            video_times.append(time.time() - t_video)
            continue

        instruction = instruction_map.get(os.path.basename(video_path), "")
        prompt = build_multi_metric_prompt(instruction)

        try:
            raw_text = run_qwen_vl(model, processor, prompt, frames)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = os.path.join(tmp_dir, f"{Path(video_path).stem}_{timestamp}.json")
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({"raw_response": raw_text}, f, ensure_ascii=False, indent=2)
            item["raw_response_file"] = raw_file

            parsed = safe_parse_scores(raw_text)
            if parsed is None:
                item["error"] = "parse_failed"
            else:
                metrics = normalize_metrics(parsed)
                item["metrics"] = metrics
        except Exception as e:
            item["error"] = str(e)

        elapsed = time.time() - t_video
        video_times.append(elapsed)
        avg_time = sum(video_times) / len(video_times)
        remaining = avg_time * (total_videos - len(video_times))
        pbar.set_postfix_str(f"{elapsed:.1f}s/vid  avg={avg_time:.1f}s  ETA={remaining/60:.0f}min")
        results.append(item)
    pbar.close()

    t_total = time.time() - t_total_start
    n_ok = sum(1 for r in results if r["error"] is None)
    n_err = sum(1 for r in results if r["error"] is not None)
    print(f"\n{'='*60}")
    print(f"  Total time : {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Videos     : {total_videos} (success={n_ok}, error={n_err})")
    if video_times:
        print(f"  Per video  : avg={sum(video_times)/len(video_times):.1f}s  "
              f"min={min(video_times):.1f}s  max={max(video_times):.1f}s")
    print(f"{'='*60}\n")

    # Save final aggregated results
    out_dir = os.path.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_shard{shard_id}" if num_shards > 1 else ""
    out_file = os.path.join(out_dir, f"{model_name}_summary_val_all_intern{suffix}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Clean up temporary directory
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--output_root", default="output_VLM")
    parser.add_argument("--tmp_root", default="tmp_VLM")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--config_path", default="video_quality/config/config.yaml")
    parser.add_argument("--max_videos", type=int, default=0, help="Only evaluate first N videos (0 = all)")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    # load model path from config
    model_path = None
    if args.config_path and os.path.exists(args.config_path):
        try:
            with open(args.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            model_path = cfg.get("ckpt", {}).get("vlm_model", None)
        except Exception:
            model_path = None

    out_file = vlm_judge(
        model_name=args.model_name,
        video_dir=args.video_dir,
        summary_json=args.summary_json,
        output_root=args.output_root,
        tmp_root=args.tmp_root,
        metrics_filter=args.metrics,
        num_frames=args.num_frames,
        model_path=model_path,
        max_videos=args.max_videos,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
