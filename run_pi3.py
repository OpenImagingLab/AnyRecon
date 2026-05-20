import torch
import argparse
import numpy as np
import cv2
import math
import os
import random
import csv
import json
import fcntl
from PIL import Image
import subprocess
from Pi3.pi3.utils.basic import load_images_as_tensor, write_ply
from geometry_utils import depth_edge, recover_focal_shift, intrinsics_from_focal_center
from Pi3.pi3.models.pi3 import Pi3
from torchvision import transforms as TF
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation, Slerp
import glob


# --- Rendering Utils ---
from numba import njit  # type: ignore
NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _splat_over_blend_numba(u_sorted, v_sorted, rgb_sorted, width, height):
        color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        alpha_buffer = np.zeros((height, width), dtype=np.float32)
        n = u_sorted.shape[0]
        for idx in range(n):
            u_coord = u_sorted[idx]
            v_coord = v_sorted[idx]
            u_floor = int(math.floor(u_coord))
            v_floor = int(math.floor(v_coord))
            if not (0 <= u_floor < width - 1 and 0 <= v_floor < height - 1):
                continue

            u_frac = u_coord - u_floor
            v_frac = v_coord - v_floor

            w00 = (1.0 - u_frac) * (1.0 - v_frac)
            w10 = u_frac * (1.0 - v_frac)
            w01 = (1.0 - u_frac) * v_frac
            w11 = u_frac * v_frac

            r = float(rgb_sorted[idx, 0])
            g = float(rgb_sorted[idx, 1])
            b = float(rgb_sorted[idx, 2])

            if w00 > 1e-8:
                a_old = alpha_buffer[v_floor, u_floor]
                a_new = w00
                one_minus_a_new = 1.0 - a_new
                color_buffer[v_floor, u_floor, 0] = r * a_new + color_buffer[v_floor, u_floor, 0] * one_minus_a_new
                color_buffer[v_floor, u_floor, 1] = g * a_new + color_buffer[v_floor, u_floor, 1] * one_minus_a_new
                color_buffer[v_floor, u_floor, 2] = b * a_new + color_buffer[v_floor, u_floor, 2] * one_minus_a_new
                alpha_buffer[v_floor, u_floor] = a_new + a_old * (1.0 - a_new)

            if w10 > 1e-8:
                uu = u_floor + 1
                a_old = alpha_buffer[v_floor, uu]
                a_new = w10
                one_minus_a_new = 1.0 - a_new
                color_buffer[v_floor, uu, 0] = r * a_new + color_buffer[v_floor, uu, 0] * one_minus_a_new
                color_buffer[v_floor, uu, 1] = g * a_new + color_buffer[v_floor, uu, 1] * one_minus_a_new
                color_buffer[v_floor, uu, 2] = b * a_new + color_buffer[v_floor, uu, 2] * one_minus_a_new
                alpha_buffer[v_floor, uu] = a_new + a_old * (1.0 - a_new)

            if w01 > 1e-8:
                vv = v_floor + 1
                a_old = alpha_buffer[vv, u_floor]
                a_new = w01
                one_minus_a_new = 1.0 - a_new
                color_buffer[vv, u_floor, 0] = r * a_new + color_buffer[vv, u_floor, 0] * one_minus_a_new
                color_buffer[vv, u_floor, 1] = g * a_new + color_buffer[vv, u_floor, 1] * one_minus_a_new
                color_buffer[vv, u_floor, 2] = b * a_new + color_buffer[vv, u_floor, 2] * one_minus_a_new
                alpha_buffer[vv, u_floor] = a_new + a_old * (1.0 - a_new)

            if w11 > 1e-8:
                uu = u_floor + 1
                vv = v_floor + 1
                a_old = alpha_buffer[vv, uu]
                a_new = w11
                one_minus_a_new = 1.0 - a_new
                color_buffer[vv, uu, 0] = r * a_new + color_buffer[vv, uu, 0] * one_minus_a_new
                color_buffer[vv, uu, 1] = g * a_new + color_buffer[vv, uu, 1] * one_minus_a_new
                color_buffer[vv, uu, 2] = b * a_new + color_buffer[vv, uu, 2] * one_minus_a_new
                alpha_buffer[vv, uu] = a_new + a_old * (1.0 - a_new)

        return color_buffer, alpha_buffer


def render_point_cloud_to_memory(
    points_3d, points_rgb, extrinsics, intrinsics, image_size, points_source_indices=None
):
    """Render a point cloud into multiple camera views.

    Returns rendered RGB frames, alpha masks, and (optionally) a per-frame
    dict mapping source frame index -> number of contributing points.
    """
    num_cameras = extrinsics.shape[0]
    width, height = image_size

    rendered_frames = []
    rendered_masks = []
    rendered_frame_source_indices = []

    if points_rgb.dtype != np.uint8:
        prgb = points_rgb.astype(np.float32)
        if prgb.max() <= 1.0:
            prgb = prgb * 255.0
        points_rgb = np.clip(prgb, 0, 255).astype(np.uint8)

    if points_source_indices is not None:
        if torch.is_tensor(points_source_indices):
            points_source_indices = points_source_indices.cpu().numpy()
        points_source_indices = points_source_indices.flatten()

    points_3d_homo = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1))], axis=1
    )

    def _render_splat(u_sorted, v_sorted, rgb_sorted, width, height):
        if NUMBA_AVAILABLE and u_sorted.size > 0:
            color_buffer, alpha_2d = _splat_over_blend_numba(
                u_sorted.astype(np.float32),
                v_sorted.astype(np.float32),
                rgb_sorted.astype(np.float32),
                width,
                height,
            )
            alpha_buffer = alpha_2d[:, :, None].astype(np.float32)
        else:
            color_buffer = np.zeros((height, width, 3), dtype=np.float32)
            alpha_buffer = np.zeros((height, width, 1), dtype=np.float32)

            for u_coord, v_coord, rgb in zip(u_sorted, v_sorted, rgb_sorted):
                u_floor = math.floor(u_coord)
                v_floor = math.floor(v_coord)

                if not (0 <= u_floor < width - 1 and 0 <= v_floor < height - 1):
                    continue

                u_frac = u_coord - u_floor
                v_frac = v_coord - v_floor

                weights = {
                    (v_floor, u_floor): (1 - u_frac) * (1 - v_frac),
                    (v_floor, u_floor + 1): u_frac * (1 - v_frac),
                    (v_floor + 1, u_floor): (1 - u_frac) * v_frac,
                    (v_floor + 1, u_floor + 1): u_frac * v_frac
                }

                for (v0, u0), alpha in weights.items():
                    if alpha > 1e-8:
                        color_buffer[v0, u0] = rgb * alpha + color_buffer[v0, u0] * (1 - alpha)
                        alpha_buffer[v0, u0] = alpha + alpha_buffer[v0, u0] * (1 - alpha)

        safe_alpha = np.maximum(alpha_buffer, 1e-6)
        color_buffer = color_buffer / safe_alpha
        img_array = np.clip(color_buffer, 0, 255).astype(np.uint8)
        return img_array, alpha_buffer

    for i in range(num_cameras):
        E = extrinsics[i]
        K = intrinsics[i]

        if E.shape == (4, 4):
            E_cam_from_world = E
        else:
            E_cam_from_world = np.eye(4)
            E_cam_from_world[:3, :4] = E

        points_cam = (E_cam_from_world @ points_3d_homo.T).T

        front_of_camera_mask = points_cam[:, 2] > 0.1
        points_cam_front = points_cam[front_of_camera_mask]
        points_rgb_front = points_rgb[front_of_camera_mask]

        if points_source_indices is not None:
            points_source_indices_front = points_source_indices[front_of_camera_mask]

        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        mask_array = np.zeros((height, width), dtype=np.uint8)
        current_frame_sources = {}

        if len(points_cam_front) > 0:
            points_img_homo = (K @ points_cam_front[:, :3].T).T
            depth = points_img_homo[:, 2]
            u = points_img_homo[:, 0] / (depth + 1e-8)
            v = points_img_homo[:, 1] / (depth + 1e-8)

            in_bounds_mask = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)

            u_in = u[in_bounds_mask]
            v_in = v[in_bounds_mask]
            rgb_in = points_rgb_front[in_bounds_mask]
            depth_in = depth[in_bounds_mask]

            if points_source_indices is not None:
                source_indices_in = points_source_indices_front[in_bounds_mask]

            non_black_mask = (rgb_in.sum(axis=1) > 0)
            u_in = u_in[non_black_mask]
            v_in = v_in[non_black_mask]
            rgb_in = rgb_in[non_black_mask]
            depth_in = depth_in[non_black_mask]

            if points_source_indices is not None:
                source_indices_in = source_indices_in[non_black_mask]
                unique, counts = np.unique(source_indices_in, return_counts=True)
                current_frame_sources = dict(zip(unique.tolist(), counts.tolist()))

            sort_indices = np.argsort(depth_in)[::-1]
            u_sorted = u_in[sort_indices]
            v_sorted = v_in[sort_indices]
            rgb_sorted = rgb_in[sort_indices]

            img_array, alpha_buffer = _render_splat(u_sorted, v_sorted, rgb_sorted, width, height)
            mask_array = (alpha_buffer.squeeze() > 0.01).astype(np.uint8) * 255

        rendered_frame_source_indices.append(current_frame_sources)

        final_frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        final_mask = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)

        rendered_frames.append(final_frame)
        rendered_masks.append(final_mask)

    return rendered_frames, rendered_masks, rendered_frame_source_indices


def save_video(frames, output_video_path, fps=10):
    """Encode a list of BGR frames to an mp4 via ffmpeg."""
    if not frames:
        print(f"Error: No frames to save for {output_video_path}")
        return False

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)

    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p',
        output_video_path
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    for frame in frames:
        if frame is None:
            continue
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        proc.stdin.write(frame.tobytes())

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"FFmpeg error while creating video {output_video_path}:\n{stderr.decode('utf-8')}")
        return False
    return True


def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


# --- Data Loading Utils ---

def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def list_directory(directory_path: str) -> list:
    if os.path.isdir(directory_path):
        files = sorted(glob.glob(os.path.join(directory_path, "*")))
        return [os.path.basename(f) for f in files]
    return []


def center_crop_to_size(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop a frame; if smaller than target, resize up first."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w < target_w or h < target_h:
        scale = max(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        h, w = frame.shape[:2]
    x1 = max(0, (w - target_w) // 2)
    y1 = max(0, (h - target_h) // 2)
    return frame[y1:y1 + target_h, x1:x1 + target_w]


def write_scene_metadata_to_file(scene_metadata, metadata_file_path):
    """Append scene metadata to CSV with file locking for multi-process safety."""
    if not scene_metadata:
        return

    try:
        os.makedirs(os.path.dirname(metadata_file_path) or '.', exist_ok=True)
        file_exists = os.path.isfile(metadata_file_path)
        mode = 'r+' if file_exists else 'a+'

        with open(metadata_file_path, mode, newline='') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            existing_file_names = set()
            if file_exists:
                try:
                    f.seek(0)
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        existing_file_names = set(row.get('file_name', '') for row in reader)
                except Exception as e:
                    print(f"Warning: Could not read existing metadata file: {e}")

            fieldnames = ['file_name', 'text', 'mask_index']
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)

            current_pos = f.tell()
            if not file_exists or current_pos == 0:
                f.seek(0)
                writer.writeheader()
            f.seek(0, os.SEEK_END)

            new_records = [r for r in scene_metadata
                           if r.get('file_name') not in existing_file_names]

            if new_records:
                writer.writerows(new_records)
                f.flush()
                print(f"Appended {len(new_records)} metadata records to {metadata_file_path}")

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error writing metadata to file: {e}")


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def _load_scene_frames(scene_dir):
    """Load all frames from a scene (video file or directory of images)."""
    frames = []
    if os.path.isfile(scene_dir) and scene_dir.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(scene_dir)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    else:
        image_dir = os.path.join(scene_dir, "images_undistort")
        files = list_directory(image_dir)
        if not files:
            image_dir = scene_dir
            files = list_directory(image_dir)
        paths = [os.path.join(image_dir, p) for p in files
                 if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for path in paths:
            img = load_image(path)
            if img is not None:
                frames.append(img)
    return frames


def _compute_target_resolution(pil_image, pixel_limit=255000, patch=14):
    """Pick the largest HxW that is patch-aligned and below the pixel budget."""
    W_orig, H_orig = pil_image.size
    if W_orig * H_orig == 0:
        return patch, patch
    scale = math.sqrt(pixel_limit / (W_orig * H_orig))
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / patch), round(H_target / patch)
    while (k * patch) * (m * patch) > pixel_limit:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch


def _smooth_render_trajectory(poses_c2w, window_size=5, rot_iters=5):
    """Smooth the rendered-frame portion of the trajectory.

    Translations use a moving average; rotations use iterative quaternion
    averaging with sign alignment to take the shortest arc.
    """
    rotations = Rotation.from_matrix(poses_c2w[:, :3, :3])
    translations = poses_c2w[:, :3, 3]

    pad_width = window_size // 2
    kernel = np.ones(window_size) / window_size
    trans_padded = np.pad(translations, ((pad_width, pad_width), (0, 0)), mode='edge')
    smooth_trans = np.stack(
        [np.convolve(trans_padded[:, k], kernel, mode='valid') for k in range(3)],
        axis=-1,
    )

    quats = rotations.as_quat()
    smooth_quats = np.copy(quats)
    for _ in range(rot_iters):
        new_quats = np.copy(smooth_quats)
        for i in range(1, len(quats) - 1):
            q_prev = smooth_quats[i - 1]
            q_curr = smooth_quats[i]
            q_next = smooth_quats[i + 1]
            if np.dot(q_curr, q_prev) < 0:
                q_prev = -q_prev
            if np.dot(q_curr, q_next) < 0:
                q_next = -q_next
            q_avg = 0.25 * q_prev + 0.5 * q_curr + 0.25 * q_next
            new_quats[i] = q_avg / np.linalg.norm(q_avg)
        smooth_quats = new_quats

    smooth_rots = Rotation.from_quat(smooth_quats)
    smooth_poses_c2w = np.tile(np.eye(4), (len(poses_c2w), 1, 1))
    smooth_poses_c2w[:, :3, :3] = smooth_rots.as_matrix()
    smooth_poses_c2w[:, :3, 3] = smooth_trans
    return smooth_poses_c2w


def process_scene(model, device, scene_dir, args):
    print(f"\n====================================================\n"
          f"Processing scene: {scene_dir}\n"
          f"====================================================")

    loaded_images_np = _load_scene_frames(scene_dir)
    if not loaded_images_np:
        print(f"Warning: No frames found in {scene_dir}. Skipping.")
        return False, []

    num_frames_total = len(loaded_images_np)
    start_time = time.time()
    print(f"--- Processing {num_frames_total} frames ---")
    batch_output_suffix = f"batch_0_{num_frames_total-1}"

    loaded_images_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in loaded_images_np]
    TARGET_W, TARGET_H = _compute_target_resolution(loaded_images_pil[0])

    imgs_list = []
    to_tensor = TF.ToTensor()
    resized_loaded_images_np = []
    for j, img_pil in enumerate(loaded_images_pil):
        resized_pil = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        imgs_list.append(to_tensor(resized_pil))
        resized_loaded_images_np.append(
            cv2.resize(loaded_images_np[j], (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        )
    imgs = torch.stack(imgs_list, dim=0).to(device)  # (B, 3, H, W)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # (1, B, 3, H, W)

    masks_batched = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks_batched = torch.logical_and(masks_batched, non_edge)  # (1, B, H, W)
    masks = masks_batched[0]  # (B, H, W)

    intrinsics_pred = None
    try:
        points = res["local_points"]
        focal, shift = recover_focal_shift(points, masks_batched, downsample_size=(64, 64))

        original_height, original_width = points.shape[-3:-1]
        aspect_ratio = original_width / original_height
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * original_width
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * original_height
        cx = original_width / 2.0
        cy = original_height / 2.0
        intrinsics_pred = intrinsics_from_focal_center(fx, fy, cx, cy)
        res["intrinsics"] = intrinsics_pred
        res["shift"] = shift
        res["camera_poses_cw"] = torch.linalg.inv(res["camera_poses"])
    except Exception as e:
        print(f"Intrinsics prediction failed: {e}")

    num_cond_frames = args.num_cond_frames
    masks_first_cond = masks.clone()
    masks_first_cond[num_cond_frames:] = False
    valid_points = res['points'][0][masks_first_cond].cpu().numpy()
    valid_colors = (imgs.permute(0, 2, 3, 1)[masks_first_cond].cpu().numpy() * 255).astype(np.uint8)

    batch_indices_map = torch.arange(imgs.shape[0], device=device).view(-1, 1, 1).expand(masks.shape)
    valid_source_indices = batch_indices_map[masks_first_cond].cpu().numpy()

    poses_c2w = res['camera_poses'][0]
    extrinsics_w2c = torch.linalg.inv(poses_c2w).cpu().numpy()
    target_extrinsics = extrinsics_w2c

    if num_frames_total > num_cond_frames + 2:
        try:
            render_poses_c2w = poses_c2w[num_cond_frames:].cpu().numpy()
            smooth_render_poses_c2w = _smooth_render_trajectory(render_poses_c2w)
            target_extrinsics[num_cond_frames:] = np.linalg.inv(smooth_render_poses_c2w)
            print(f"Smoothed camera trajectory for {len(render_poses_c2w)} rendered frames.")
        except Exception as e:
            print(f"Warning: Camera smoothing failed: {e}")

    H_img, W_img = imgs.shape[-2:]
    if intrinsics_pred is not None:
        target_intrinsics = intrinsics_pred[0].cpu().numpy()
    else:
        fx, fy = W_img * 1.0, W_img * 1.0
        cx, cy = W_img / 2.0, H_img / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        target_intrinsics = np.stack([K] * num_frames_total, axis=0)

    rendered_frames, rendered_masks, rendered_frame_source_indices = render_point_cloud_to_memory(
        points_3d=valid_points,
        points_rgb=valid_colors,
        extrinsics=target_extrinsics,
        intrinsics=target_intrinsics,
        image_size=(W_img, H_img),
        points_source_indices=valid_source_indices,
    )

    FINAL_W, FINAL_H = 896, 512
    video_filename = f"{os.path.basename(scene_dir)}_{batch_output_suffix}"
    white_mask_3c = np.full((FINAL_H, FINAL_W, 3), 255, dtype=np.uint8)

    # Camera JSON: per-frame pose + dominant source frames for render views
    camera_json_path = os.path.join(args.camera_dir, f"{video_filename}.json")
    camera_data = []
    for idx in range(num_cond_frames):
        camera_data.append({
            "frame_id": idx,
            "is_condition": True,
            "source_frames": [idx],
        })
    for idx in range(num_cond_frames, num_frames_total):
        if args.use_retrieval:
            source_stats = rendered_frame_source_indices[idx] if rendered_frame_source_indices else {}
            top_sources = sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            sources = sorted(int(k) for k, _ in top_sources)
        else:
            sources = list(range(num_cond_frames))
        camera_data.append({
            "frame_id": idx,
            "is_condition": False,
            "source_frames": sources,
        })
    save_json(camera_data, camera_json_path)

    # Chunked video generation: group render frames into chunks of 35,
    # prepend each chunk with its top-contributing condition frames.
    render_start_idx = num_cond_frames
    render_total = num_frames_total - num_cond_frames
    if render_total > 0:
        chunk_size = 35
        num_chunks = math.ceil(render_total / chunk_size)
        top_k_per_chunk = 10

        for i in range(num_chunks):
            start = render_start_idx + i * chunk_size
            if start + chunk_size > num_frames_total:
                break
            end = min(start + chunk_size, num_frames_total)
            chunk_indices = range(start, end)

            agg_sources = {}
            for frame_idx in chunk_indices:
                if frame_idx >= len(rendered_frame_source_indices):
                    continue
                stats = rendered_frame_source_indices[frame_idx]
                top_per_frame = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]
                for src, count in top_per_frame:
                    agg_sources[src] = agg_sources.get(src, 0) + count

            top_chunk = sorted(agg_sources.items(), key=lambda x: x[1], reverse=True)[:top_k_per_chunk]
            top_chunk_indices = sorted(int(k) for k, _ in top_chunk)

            chunk_video_frames = []
            chunk_mask_frames = []
            for src_idx in top_chunk_indices:
                if 0 <= src_idx < len(resized_loaded_images_np):
                    chunk_video_frames.append(center_crop_to_size(resized_loaded_images_np[src_idx], FINAL_W, FINAL_H))
                    chunk_mask_frames.append(white_mask_3c)
            for frame_idx in chunk_indices:
                chunk_video_frames.append(center_crop_to_size(rendered_frames[frame_idx], FINAL_W, FINAL_H))
                chunk_mask_frames.append(center_crop_to_size(rendered_masks[frame_idx], FINAL_W, FINAL_H))

            chunk_suffix = f"chunk_{i}_frames_{start}_{end-1}"
            save_video(chunk_video_frames, os.path.join(args.condition_dir, f"{video_filename}_{chunk_suffix}.mp4"), fps=10)
            save_video(chunk_mask_frames, os.path.join(args.mask_dir, f"{video_filename}_{chunk_suffix}.mp4"), fps=10)

            txt_path = os.path.join(args.condition_dir, f"{video_filename}_{chunk_suffix}_info.txt")
            with open(txt_path, 'w') as f:
                f.write(f"Condition Frame Count: {len(top_chunk_indices)}\n")
                f.write(f"Render Frame Count: {len(chunk_indices)}\n")
                f.write(f"Condition Frame Indices: {top_chunk_indices}\n")

    scene_metadata = [{
        "file_name": f"{video_filename}.mp4",
        "text": "",
        "mask_index": ",".join(map(str, range(num_cond_frames))),
    }]

    print(f"--- Scene processed in {time.time() - start_time:.2f} seconds ---")
    return True, scene_metadata


def main():
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model (Batch Processing).")

    parser.add_argument("--base_scene_dir", type=str, default="DL3DV/DL3DV-10K/testset",
                        help="Base directory where scene folders are located.")
    parser.add_argument("--ckpt", type=str, default="./Pi3/model.safetensors",
                        help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/petrelfs/chenyutian/project/data_pipeline/floor7_long_smooth35",
                        help="Base directory containing all data subdirectories")
    parser.add_argument("--num_cond_frames", type=int, default=6,
                        help="Number of leading frames used as conditioning views.")
    parser.add_argument("--use_retrieval", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable geometry-aware retrieval to pick top contributing "
                             "condition frames per chunk. If disabled (default), every "
                             "chunk uses all conditioning frames as-is.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    args.train_dir = os.path.join(args.output_dir, "train")
    args.condition_dir = os.path.join(args.output_dir, "condition")
    args.mask_dir = os.path.join(args.output_dir, "mask")
    args.camera_dir = os.path.join(args.output_dir, "cameras")
    args.metadata_file = os.path.join(args.output_dir, "metadata.csv")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    ensure_dir(args.train_dir)
    ensure_dir(args.condition_dir)
    ensure_dir(args.mask_dir)
    ensure_dir(args.camera_dir)
    ensure_dir(os.path.dirname(args.metadata_file))

    if not os.path.exists(args.base_scene_dir):
        print(f"Error: Path not found at {args.base_scene_dir}")
        return

    if os.path.isfile(args.base_scene_dir):
        all_scenes = [args.base_scene_dir]
    elif (os.path.exists(os.path.join(args.base_scene_dir, "images_undistort"))
          or os.path.exists(os.path.join(args.base_scene_dir, "images"))):
        all_scenes = [args.base_scene_dir]
    else:
        all_scenes = [os.path.join(args.base_scene_dir, d)
                      for d in os.listdir(args.base_scene_dir)
                      if os.path.isfile(os.path.join(args.base_scene_dir, d))]
        if not all_scenes:
            all_scenes = [args.base_scene_dir]

    print(f"Found {len(all_scenes)} scenes to process.")

    for scene_dir in tqdm(all_scenes, desc="Processing Scenes"):
        success, scene_metadata = process_scene(model, device, scene_dir, args)
        if success:
            if scene_metadata:
                write_scene_metadata_to_file(scene_metadata, args.metadata_file)
        else:
            print(f"Failed to process {scene_dir}")

    print("Finished.")


if __name__ == '__main__':
    main()
