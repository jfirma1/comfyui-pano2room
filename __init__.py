"""
ComfyUI Custom Node: Pano2Room
Wraps the Pano2Room pipeline (SIGGRAPH Asia 2024) as a ComfyUI node.
Converts a single 360° panorama into a walkable 3D Gaussian Splat.

Installation:
1. Clone Pano2Room into ComfyUI/custom_nodes/:
   cd ComfyUI/custom_nodes
   git clone https://github.com/TrickyGo/Pano2Room.git comfyui-pano2room
2. Copy THIS file into the cloned folder as __init__.py (or nodes.py)
   (Alternatively, place this file alongside pano2room.py)
3. Install Pano2Room dependencies (see Pano2Room README)
4. Download checkpoints (see Pano2Room checkpoints/README.md)
5. Restart ComfyUI

The node expects the Pano2Room repo structure to be intact:
  comfyui-pano2room/
    __init__.py          (this file)
    pano2room.py         (original Pano2Room entry point)
    modules/             (Pano2Room modules)
    gaussian_renderer/   (GS renderer)
    scene/               (scene/gaussian model)
    utils/               (Pano2Room utils)
    checkpoints/         (pretrained weights)
    input/               (working input dir)
    output/              (working output dir)
"""

import os
import sys
import shutil
import glob
import subprocess
import numpy as np
import torch
from PIL import Image

# Resolve paths relative to THIS file
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PANO2ROOM_DIR = NODE_DIR  # This file lives inside the Pano2Room repo clone

# ---------------------------------------------------------------------------
# Helper: tensor <-> PIL conversions (ComfyUI IMAGE tensors are BHWC float32)
# ---------------------------------------------------------------------------
def tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor [B,H,W,C] to PIL Image (first frame)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_img):
    """Convert PIL Image to ComfyUI IMAGE tensor [1,H,W,C]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr).unsqueeze(0)


def load_images_from_dir(directory, max_images=None):
    """Load all PNG/JPG images from a directory as a batched tensor."""
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(os.path.join(directory, pat))))
    if max_images:
        paths = paths[:max_images]
    
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(pil_to_tensor(img))
    
    if tensors:
        return torch.cat(tensors, dim=0)
    return None


# ---------------------------------------------------------------------------
# Node 1: Pano2Room — Full pipeline (panorama → 3DGS)
# ---------------------------------------------------------------------------
class Pano2RoomNode:
    """
    Converts a single 360° equirectangular panorama into a 3D Gaussian Splat
    using the Pano2Room pipeline (SIGGRAPH Asia 2024).
    
    Input: IMAGE (equirectangular panorama)
    Outputs: 
        - GS_PLY_PATH: Path to the output .ply Gaussian Splat file
        - RENDERED_VIEWS: Batch of rendered novel view images
        - MESH_PATH: Path to the output mesh file
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
            },
            "optional": {
                "run_sdft": ("BOOLEAN", {"default": False, 
                    "tooltip": "Fine-tune Stable Diffusion inpainter on this specific panorama before running. Improves quality but adds ~10 min."}),
                "sdft_training_steps": ("INT", {"default": 500, "min": 100, "max": 2000, "step": 50,
                    "tooltip": "How long the AI trains on your room's lighting. Higher = more accurate textures, but risks baking in artifacts."}),
                "inpaint_stride": ("INT", {"default": 20, "min": 1, "max": 60, "step": 1,
                    "tooltip": "Lower number = more holes patched but slower. 20 is fast, 5 is meticulous, 1 checks every frame."}),
                "render_resolution": (["512", "768", "1024"], {"default": "512"}),
                "camera_fov": ("FLOAT", {"default": 90.0, "min": 45.0, "max": 150.0, "step": 1.0,
                    "tooltip": "Field of view for the internal patch camera."}),
                "room_scale": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Physical scale of the room. <1.0 = larger space, >1.0 = smaller space."}),
                "center_offset_x": ("FLOAT", {"default": -0.2, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Shifts the starting position of the camera Left/Right."}),
                "center_offset_forward": ("FLOAT", {"default": 0.3, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Shifts the starting position of the camera Forward/Back."}),
                "save_debug_passes": ("BOOLEAN", {"default": False,
                    "tooltip": "Saves raw masks and depth passes to the output folder for external VFX."}),
                "use_sky_mask": ("BOOLEAN", {"default": True,
                    "tooltip": "Use SegFormer sky masking to exclude sky from geometry (recommended for outdoor scenes). Disable for A/B testing."}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 7,
                    "tooltip": "CUDA device ID to use"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("gs_ply_path", "rendered_views", "mesh_path")
    FUNCTION = "run_pano2room"
    CATEGORY = "3D/Pano2Room"
    
    def run_pano2room(self, panorama, run_sdft=False, sdft_training_steps=500, inpaint_stride=20, render_resolution="512", camera_fov=90.0, room_scale=0.6, center_offset_x=-0.2, center_offset_forward=0.3, save_debug_passes=False, use_sky_mask=True, gpu_id=0):
        # --- Setup directories ---
        input_dir = os.path.join(PANO2ROOM_DIR, "input")
        output_dir = os.path.join(PANO2ROOM_DIR, "output")
        results_dir = os.path.join(output_dir, "Pano2Room-results")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Save input panorama ---
        pil_img = tensor_to_pil(panorama)
        input_path = os.path.join(input_dir, "input_panorama.png")
        pil_img.save(input_path)
        print(f"[Pano2Room] Saved input panorama: {input_path} "
              f"({pil_img.width}x{pil_img.height})")
        
        # --- Optionally run SDFT (fine-tune inpainter) ---
        if run_sdft:
            print(f"[Pano2Room] Running SDFT fine-tuning for {sdft_training_steps} steps...")
            # Clean previous SDFT weights if they exist
            sdft_weights_dir = os.path.join(output_dir, "SDFT_weights")
            if os.path.exists(sdft_weights_dir):
                shutil.rmtree(sdft_weights_dir)
            
            # Step 1: Create SDFT pairs
            self._run_script(
                "create_SDFT_pairs.py",
                gpu_id=gpu_id,
                desc="SDFT pair creation"
            )
            
            # Step 2: Train SDFT
            self._run_script(
                "train_SDFT.py",
                gpu_id=gpu_id,
                desc="SDFT training",
                extra_args=["--max_train_steps", str(sdft_training_steps)]
            )
        
        # --- Run Pano2Room ---
        print(f"[Pano2Room] Running Pano2Room pipeline with stride {inpaint_stride}...")
        
        # Compile all the user parameters
        p2r_args = [
            "--inpaint_stride", str(inpaint_stride),
            "--resolution", str(render_resolution),
            "--fov", str(camera_fov),
            "--room_scale", str(room_scale),
            "--offset_x", str(center_offset_x),
            "--offset_z", str(center_offset_forward)
        ]
        if not use_sky_mask:
            p2r_args.append("--disable_sky_mask")

        if save_debug_passes:
            p2r_args.append("--save_details")

        self._run_script(
            "pano2room.py",
            gpu_id=gpu_id,
            desc="Pano2Room",
            extra_args=p2r_args
        )
        
        # --- Collect outputs ---
        # Find GS PLY
        gs_ply_path = ""
        ply_candidates = glob.glob(os.path.join(results_dir, "**", "*.ply"), recursive=True)
        if not ply_candidates:
            ply_candidates = glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True)
        if ply_candidates:
            # Prefer the largest PLY (likely the GS, not sparse cloud)
            gs_ply_path = max(ply_candidates, key=os.path.getsize)
            print(f"[Pano2Room] Found GS PLY: {gs_ply_path}")
        else:
            print("[Pano2Room] WARNING: No PLY file found in output!")
        
        # Find mesh
        mesh_path = ""
        mesh_candidates = []
        for ext in ["*.obj", "*.glb", "*.gltf"]:
            mesh_candidates.extend(
                glob.glob(os.path.join(output_dir, "**", ext), recursive=True)
            )
        if mesh_candidates:
            mesh_path = mesh_candidates[0]
            print(f"[Pano2Room] Found mesh: {mesh_path}")
        
        # Load rendered novel views
        rendered_views = None
        # Look for rendered images in results
        for search_dir in [results_dir, output_dir]:
            if os.path.isdir(search_dir):
                rendered_views = load_images_from_dir(search_dir, max_images=50)
                if rendered_views is not None:
                    print(f"[Pano2Room] Loaded {rendered_views.shape[0]} rendered views")
                    break
        
        if rendered_views is None:
            # Return the input as fallback
            rendered_views = panorama
            print("[Pano2Room] No rendered views found, returning input")
        
        return (gs_ply_path, rendered_views, mesh_path)
    
    def _run_script(self, script_name, gpu_id=0, desc="", extra_args=None):
        """Run a Pano2Room Python script as a subprocess."""
        script_path = os.path.join(PANO2ROOM_DIR, script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"[Pano2Room] Script not found: {script_path}\n"
                f"Make sure the Pano2Room repo is cloned into {PANO2ROOM_DIR}"
            )
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Use the same Python interpreter that's running ComfyUI
        python_exe = sys.executable
        
        cmd = [python_exe, script_path]
        if extra_args:
            cmd.extend(extra_args)
            
        print(f"[Pano2Room] Running {desc}: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=PANO2ROOM_DIR,
            env=env,
            capture_output=True,
            text=True
        )
        
        # Print output for debugging
        if result.stdout:
            for line in result.stdout.strip().split('\n')[-20:]:
                print(f"[Pano2Room] {line}")
        
        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            print(f"[Pano2Room] ERROR in {desc}:\n{error_msg}")
            raise RuntimeError(
                f"Pano2Room {desc} failed (exit code {result.returncode}).\n"
                f"Last error output:\n{error_msg}"
            )
        
        print(f"[Pano2Room] {desc} completed successfully")


# ---------------------------------------------------------------------------
# Node 2: Pano2Room PLY Loader — Load the output GS PLY for downstream use
# ---------------------------------------------------------------------------
class Pano2RoomPLYLoader:
    """
    Load a Gaussian Splat PLY file path from Pano2Room output.
    Useful for connecting to GS viewer/renderer nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "load_ply"
    CATEGORY = "3D/Pano2Room"
    
    def load_ply(self, ply_path):
        if not os.path.exists(ply_path):
            print(f"[Pano2Room PLY Loader] WARNING: File not found: {ply_path}")
        else:
            size_mb = os.path.getsize(ply_path) / (1024 * 1024)
            print(f"[Pano2Room PLY Loader] Loaded: {ply_path} ({size_mb:.1f} MB)")
        return (ply_path,)


# ---------------------------------------------------------------------------
# Node 3: Pano2Room Camera Trajectory — Generate camera trajectory files
# ---------------------------------------------------------------------------
class Pano2RoomCameraTrajectory:
    """
    Generate a camera trajectory for rendering novel views from the 
    Pano2Room output. Creates camera extrinsic [R|T] matrices.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trajectory_type": (["orbit", "dolly_forward", "dolly_lateral", "custom_circle"],),
                "num_frames": ("INT", {"default": 60, "min": 1, "max": 600}),
                "radius": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Camera movement radius/distance in meters"}),
                "height": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Camera height offset in meters"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trajectory_dir",)
    FUNCTION = "generate_trajectory"
    CATEGORY = "3D/Pano2Room"
    
    def generate_trajectory(self, trajectory_type, num_frames, radius, height):
        traj_dir = os.path.join(PANO2ROOM_DIR, "input", "Camera_Trajectory_Custom")
        os.makedirs(traj_dir, exist_ok=True)
        
        # Clean previous trajectory
        for f in glob.glob(os.path.join(traj_dir, "*.txt")):
            os.remove(f)
        
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)
            angle = 2 * np.pi * t
            
            if trajectory_type == "orbit":
                # Orbit around the room center
                tx = radius * np.cos(angle)
                tz = radius * np.sin(angle)
                ty = height
                # Camera looks toward center
                forward = np.array([-tx, 0, -tz])
                forward = forward / (np.linalg.norm(forward) + 1e-8)
                
            elif trajectory_type == "dolly_forward":
                # Move forward along Z axis
                tx = 0.0
                tz = -radius * t  # Move forward
                ty = height
                forward = np.array([0, 0, -1])
                
            elif trajectory_type == "dolly_lateral":
                # Move left to right
                tx = radius * (t - 0.5) * 2  # -radius to +radius
                tz = 0.0
                ty = height
                forward = np.array([0, 0, -1])
                
            elif trajectory_type == "custom_circle":
                # Small circle with look-ahead
                tx = radius * 0.3 * np.cos(angle)
                tz = radius * 0.3 * np.sin(angle)
                ty = height + 0.1 * np.sin(angle * 2)
                # Look slightly ahead on the circle
                look_angle = angle + 0.3
                forward = np.array([
                    np.cos(look_angle) - tx,
                    0,
                    np.sin(look_angle) - tz
                ])
                forward = forward / (np.linalg.norm(forward) + 1e-8)
            
            # Build rotation matrix (camera looks along -Z in OpenGL convention)
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)
            
            R = np.eye(3)
            R[0, :] = right
            R[1, :] = up
            R[2, :] = -forward
            
            T = np.array([tx, ty, tz])
            
            # Build 4x4 [R|T] matrix
            RT = np.eye(4)
            RT[:3, :3] = R
            RT[:3, 3] = T
            
            # Save as text file (Pano2Room format)
            frame_path = os.path.join(traj_dir, f"{i:04d}.txt")
            np.savetxt(frame_path, RT, fmt="%.8f")
        
        print(f"[Pano2Room Camera] Generated {num_frames} frames, "
              f"type={trajectory_type}, radius={radius}m")
        return (traj_dir,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Pano2Room": Pano2RoomNode,
    "Pano2RoomPLYLoader": Pano2RoomPLYLoader,
    "Pano2RoomCameraTrajectory": Pano2RoomCameraTrajectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pano2Room": "Pano2Room (Panorama → 3DGS)",
    "Pano2RoomPLYLoader": "Pano2Room PLY Loader",
    "Pano2RoomCameraTrajectory": "Pano2Room Camera Trajectory",
}