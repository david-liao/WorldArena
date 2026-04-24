import torch
import sys

import os
from  tqdm import tqdm
import hashlib
import requests

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from .utils import load_dimension_info

import time
import json
import json_repair
from pathlib import Path


def inference(model, processor ,video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    from submodel.qwen_vl_utils import process_vision_info  # local import to avoid heavy deps when caption is unused
    messages = [
        {"role": "system", "content": "You are a helpful assistant in analyzing videos."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    # transformers processor expects a scalar fps; process_vision_info returns a list like [30]
    if isinstance(fps_inputs, (list, tuple)):
        fps_inputs = fps_inputs[0] if len(fps_inputs) > 0 else None
    # print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    start_time = time.time()
    # adding timeout to avoid long waiting time
    try:
        

        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        output_text = "Error: " + str(e)
    
    end_time = time.time()
    return output_text[0]


def prepare_prompt(video_path):
   
   prompt = """
   Analyze the video captured by the overhead camera mounted on a robotic and perform the following tasks:
   
   NOTED: There should be **no human hands** in the video. These are often wrongly generated from the robot's gripper. If human hands appear, note them as **abnormal** and a **violation of logical constraints**.

   1. Describe the video in general:
      - Provide a brief description on what tasks the robotic is performing. When something anomaly happens, pay special attention to it.

   2. Describe the events:
      - Provide a detailed, step-by-step description of all actions in the video, focusing on their sequence.

   3. Identify key events with logical constraints:
      - Extract critical actions subject to logical rules, such as:
      - Prerequisites (e.g., opening a door before accessing items).
      - Avoiding physical violations (e.g., objects passing through barriers).
      - Maintaining logical task order.
      - Identify any violations of these constraints.
      
   Provide the result in json format with each key representing a task.

   {
   "General": Brief description on tasks the robotic is performing.
   "Events": Chronological list of all actions.
   "Logical_Constraints": Key actions, their constraints, and whether they are satisfied.
   "Overall_Constraints": True if the logical constraints are satisfied, false otherwise.
   }

   """

   return prompt



def caption_reference(
                        model_name,
                        model_path,
                        video_folder_root,
                        save_path,
                        path_marker=None,
                        key_prefix=None,
                        **kwargs
                        ):
    # `path_marker` is the directory name that separates the dataset root from
    # <task>/<episode>/... inside each mp4's path. Defaults to `f"{model_name}_dataset"`,
    # which matches the legacy layout where videos live under `.../generated_dataset/...`
    # or `.../gt_dataset/...`. For sharded evaluations the videos live under `.../shard_N/...`
    # so the caller can pass `path_marker="shard_N"` to locate the boundary correctly.
    # `key_prefix` is the token that will be prepended to every output key (and used so
    # that downstream `semantic_alignment.py` can still rely on the canonical
    # `generated_dataset_` / `gt_dataset_` prefix regardless of what marker we scanned on).
    effective_path_marker = path_marker if path_marker else f"{model_name}_dataset"
    effective_key_prefix = key_prefix if key_prefix else f"{model_name}_dataset"

    # Delay heavy imports so other dimensions can run without Qwen dependencies
    from transformers import AutoProcessor, AutoModelForCausalLM
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLClass
        qwen_cls = QwenVLClass
        qwen_kwargs = {}
    except ImportError:
        # Fallback to generic loader with remote code when class is absent
        qwen_cls = AutoModelForCausalLM
        qwen_kwargs = {"trust_remote_code": True}

    model = qwen_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        **qwen_kwargs,
    )

    processor = AutoProcessor.from_pretrained(model_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    json_file_name = f"{model_name}_caption_responses.json"
    json_file_name = os.path.join(save_path, json_file_name)

    if not os.path.exists(json_file_name):

        all_mp4_files = load_dimension_info(video_folder_root, dimension='semantic_alignment')


        all_responses = {}
        error_counter = 0
        for mp4_file in tqdm(all_mp4_files):

            prompt = prepare_prompt(mp4_file)
            try:
                response = json_repair.loads(inference(model, processor, mp4_file, prompt))
            except Exception as e:
                print(f"Error: {str(e)}")
                response = "Error: " + str(e)

            parts = mp4_file.split('/')
            try:
                start_index = parts.index(effective_path_marker) + 1
                video_index = parts.index('video')
                selected_parts = parts[start_index:video_index]
                mp4_file_name = "_".join([effective_key_prefix] + selected_parts)
            except ValueError:
                # Keep each failure under a unique key so entries don't collapse and
                # swallow the whole caption response dict.
                error_counter += 1
                mp4_file_name = f"error_in_filename_construction_{error_counter}_{os.path.basename(mp4_file.rstrip('/'))}"
                print(
                    f"[caption_reference] Could not locate path_marker "
                    f"'{effective_path_marker}' or 'video' in {mp4_file}"
                )

            all_responses[mp4_file_name] = response
            
        # save to json file
        with open(json_file_name, 'w') as f:
            json.dump(all_responses, f, indent=4)

    else:
        return