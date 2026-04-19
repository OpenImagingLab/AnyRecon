import torch
from diffsynth import ModelManager, save_video, VideoData
from pipeline.anyrecon_pipeline import AnyReconPipeline
from PIL import Image
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np

parser = argparse.ArgumentParser(description="Wan Video Pipeline with custom paths")
parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing condition and mask folders")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
parser.add_argument("--is_block", action="store_true", help="Whether to use block attention")
parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint")
args = parser.parse_args()

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

model_manager.load_models(
    ["./checkpoints/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,
)

model_manager.load_models(
    [
        [
            "./checkpoints/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "./checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        "./checkpoints/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

model_manager.load_lora(args.lora_path, lora_alpha=1.0)
pipe = AnyReconPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

condition_dir = os.path.join(args.root_dir, "condition")
mask_dir = os.path.join(args.root_dir, "mask")

os.makedirs(args.output_dir, exist_ok=True)

video_files = sorted([f for f in os.listdir(condition_dir) if f.endswith('.mp4')])

previous_frames = None

for video_file in video_files:
    video_path = os.path.join(condition_dir, video_file)
    mask_pth = os.path.join(mask_dir, video_file)
    
    print(f"Processing {video_file}...")

    cap = cv2.VideoCapture(video_path)
    video = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.append(pil_frame)
    cap.release()
    
    info_txt_path = os.path.join(condition_dir, video_file.rsplit('.mp4', 1)[0] + '_info.txt')

    insert_index = None
    if os.path.exists(info_txt_path):
        try:
            with open(info_txt_path, 'r') as f:
                for line in f:
                    if "Condition Frame Count:" in line:
                        insert_index = int(line.split(":")[1].strip())
                        print(f"Read insert_index {insert_index} from {info_txt_path}")
                        break
        except Exception as e:
            print(f"Error reading info txt {info_txt_path}: {e}")

    if previous_frames is not None:
        if len(video) >= insert_index:
            video.insert(insert_index, previous_frames[0])
            video.insert(insert_index + 1, previous_frames[1])
            print(f"Inserted previous last frames at index {insert_index} and {insert_index + 1} of {video_file} (frames shifted)")
        else:
            print(f"Warning: Video {video_file} is too short to insert frame at index {insert_index}")

    image = video[0]

    cap = cv2.VideoCapture(mask_pth)
    mask_video = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        mask_video.append(pil_frame)
    cap.release()
    
    if previous_frames is not None:
         if len(mask_video) >= insert_index:
            ref_mask = mask_video[0]
            white_mask = torch.ones_like(ref_mask) * 255
            mask_video.insert(insert_index, white_mask)
            mask_video.insert(insert_index + 1, white_mask)
            print(f"Inserted white masks at index {insert_index} and {insert_index + 1} of {video_file} (frames shifted)")

    if args.is_block:
        num_frames = len(video) // 8 * 8 
    else:
        num_frames = len(video)

    video = video[:num_frames]
    mask_video = mask_video[:num_frames]
    video_output = pipe(
        prompt=" ",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        input_video=video,
        num_inference_steps=4,
        seed=1, num_frames=num_frames, tiled=True,
        height=512, width=896,
        mask_indices=[0], 
        mask_frames=mask_video,
        is_block=args.is_block
    )
    
    root_dir_name = os.path.basename(os.path.normpath(args.root_dir))
    output_save_path = os.path.join(args.output_dir, f'{video_file}')
    save_video(video_output, output_save_path, fps=10, quality=9, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    print(f"Saved result to {output_save_path}")

    if len(video_output) >= 2:
        previous_frames = video_output[-2:]
