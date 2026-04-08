#!/bin/bash
# =============================================================================
# Pano2Room Installation Script
# Target: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# =============================================================================
set -e

echo "============================================="
echo "Pano2Room Installer for RunPod Container"
echo "PyTorch 2.4 / Python 3.11 / CUDA 12.4"
echo "============================================="

# --- Config ---
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"
NODE_DIR="${COMFYUI_DIR}/custom_nodes/comfyui-pano2room"

# --- Step 1: Clone Pano2Room ---
echo ""
echo "[1/7] Cloning Pano2Room..."
if [ -d "$NODE_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$NODE_DIR" && git pull || true
else
    git clone https://github.com/TrickyGo/Pano2Room.git "$NODE_DIR"
fi
cd "$NODE_DIR"

# --- Step 2: Install Python requirements ---
echo ""
echo "[2/7] Installing Python requirements..."
pip install --break-system-packages -q \
    opencv-python \
    open3d \
    trimesh \
    timm \
    h5py \
    kornia \
    "albumentations==0.5.2" \
    webdataset \
    omegaconf \
    easydict \
    pytorch-lightning \
    icecream \
    plyfile \
    transformers \
    "tifffile==2023.7.10" \
    "imageio==2.31.5" \
    "imageio-ffmpeg==0.4.7" \
    datasets \
    peft \
    xformers \
    diffusers \
    accelerate

# --- Step 3: Install compiled CUDA packages ---
echo ""
echo "[3/7] Installing compiled CUDA packages (this takes a few minutes)..."

# simple-knn
echo "  Installing simple-knn..."
pip install --break-system-packages -q \
    git+https://github.com/camenduru/simple-knn.git 2>/dev/null || \
    echo "  WARNING: simple-knn failed — will use KDTree fallback"

# diff-gaussian-rasterization with depth support
echo "  Installing diff-gaussian-rasterization-w-depth..."
pip install --break-system-packages -q \
    git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git

# pytorch3d
echo "  Installing pytorch3d (this is slow, ~5 min)..."
# Try prebuilt wheel first for speed
pip install --break-system-packages -q \
    "pytorch3d>=0.7.0" 2>/dev/null || \
    pip install --break-system-packages -q \
    git+https://github.com/facebookresearch/pytorch3d.git

# tiny-cuda-nn
echo "  Installing tiny-cuda-nn..."
pip install --break-system-packages -q \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 2>/dev/null || \
    echo "  WARNING: tiny-cuda-nn failed — may not be needed for base pipeline"

# --- Step 4: Apply known fixes ---
echo ""
echo "[4/7] Applying known compatibility fixes..."

# Fix 1: gaussian_renderer import
RENDERER_INIT="${NODE_DIR}/gaussian_renderer/__init__.py"
if [ -f "$RENDERER_INIT" ]; then
    # Replace the broken import
    sed -i 's/from depth_diff_gaussian_rasterization_min import/from diff_gaussian_rasterization import/' "$RENDERER_INIT"
    # Fix debug parameter
    sed -i 's/debug=opt.debug/debug=False/' "$RENDERER_INIT"
    echo "  Fixed gaussian_renderer imports"
fi

# Fix 2: simple_knn fallback in gaussian_model.py
GAUSSIAN_MODEL="${NODE_DIR}/scene/gaussian_model.py"
if [ -f "$GAUSSIAN_MODEL" ]; then
    # Check if simple_knn import exists and add fallback
    if grep -q "from simple_knn._C import distCUDA2" "$GAUSSIAN_MODEL"; then
        sed -i 's/from simple_knn._C import distCUDA2/try:\n    from simple_knn._C import distCUDA2\nexcept ImportError:\n    from scipy.spatial import KDTree\n    import numpy as np_fallback\n    def distCUDA2(points):\n        pts = points.detach().cpu().float().numpy()\n        dists, _ = KDTree(pts).query(pts, k=4)\n        mean_d = (dists[:, 1:] ** 2).mean(1)\n        return torch.tensor(mean_d, dtype=points.dtype, device=points.device)/' "$GAUSSIAN_MODEL" 2>/dev/null || \
        echo "  NOTE: Could not auto-fix simple_knn — may need manual edit"
    fi
    echo "  Applied simple_knn fallback"
fi

# --- Step 5: Download checkpoints ---
echo ""
echo "[5/7] Downloading checkpoints..."
CKPT_DIR="${NODE_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# Check if checkpoints README has download links
CKPT_README="${CKPT_DIR}/README.md"
if [ -f "$CKPT_README" ]; then
    echo "  Checkpoint README found. Checking for existing weights..."
    
    # Count existing checkpoint files
    CKPT_COUNT=$(find "$CKPT_DIR" -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" 2>/dev/null | wc -l)
    
    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "  Found $CKPT_COUNT checkpoint files — skipping download"
    else
        echo ""
        echo "  ============================================="
        echo "  MANUAL STEP REQUIRED: Download checkpoints"
        echo "  ============================================="
        echo "  Read: ${CKPT_README}"
        echo "  Download the checkpoint files from the links"
        echo "  in that README and place them in:"
        echo "  ${CKPT_DIR}/"
        echo "  ============================================="
        echo ""
        # Try to cat the README for convenience
        cat "$CKPT_README" 2>/dev/null || true
    fi
else
    echo "  No checkpoint README found — check Pano2Room repo for download links"
fi

# --- Step 6: Copy ComfyUI node wrapper ---
echo ""
echo "[6/7] Installing ComfyUI node wrapper..."

# Check if __init__.py already has our node code
if grep -q "Pano2RoomNode" "${NODE_DIR}/__init__.py" 2>/dev/null; then
    echo "  ComfyUI node wrapper already installed"
else
    # If user has the wrapper file nearby, copy it
    WRAPPER_CANDIDATES=(
        "/mnt/user-data/outputs/__init__.py"
        "/workspace/__init__.py"
        "$(dirname "$0")/__init__.py"
    )
    
    FOUND_WRAPPER=""
    for candidate in "${WRAPPER_CANDIDATES[@]}"; do
        if [ -f "$candidate" ] && grep -q "Pano2RoomNode" "$candidate" 2>/dev/null; then
            FOUND_WRAPPER="$candidate"
            break
        fi
    done
    
    if [ -n "$FOUND_WRAPPER" ]; then
        cp "$FOUND_WRAPPER" "${NODE_DIR}/__init__.py"
        echo "  Copied ComfyUI wrapper from: $FOUND_WRAPPER"
    else
        echo "  WARNING: ComfyUI wrapper (__init__.py) not found."
        echo "  Please copy the Pano2Room ComfyUI node file to:"
        echo "  ${NODE_DIR}/__init__.py"
    fi
fi

# --- Step 7: Verify installation ---
echo ""
echo "[7/7] Verifying installation..."
cd "$NODE_DIR"

python -c "
import sys
errors = []

try:
    import torch
    print(f'  PyTorch {torch.__version__} — CUDA {torch.version.cuda} — GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
except Exception as e:
    errors.append(f'PyTorch: {e}')

try:
    import pytorch3d
    print(f'  pytorch3d OK')
except Exception as e:
    errors.append(f'pytorch3d: {e}')

try:
    from diff_gaussian_rasterization import GaussianRasterizer
    print(f'  diff-gaussian-rasterization OK')
except Exception as e:
    errors.append(f'diff-gaussian-rasterization: {e}')

try:
    import kornia
    print(f'  kornia OK')
except Exception as e:
    errors.append(f'kornia: {e}')

try:
    import open3d
    print(f'  open3d OK')
except Exception as e:
    errors.append(f'open3d: {e}')

try:
    import diffusers
    print(f'  diffusers OK')
except Exception as e:
    errors.append(f'diffusers: {e}')

try:
    import tinycudann
    print(f'  tiny-cuda-nn OK')
except Exception as e:
    errors.append(f'tiny-cuda-nn: {e} (may not be required)')

if errors:
    print()
    print('  ISSUES:')
    for e in errors:
        print(f'    ✗ {e}')
else:
    print()
    print('  All dependencies verified!')
" 2>&1

echo ""
echo "============================================="
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints (if not done above)"
echo "  2. Test standalone:  cd ${NODE_DIR} && bash scripts/run_Pano2Room.sh"
echo "  3. Restart ComfyUI to load the new nodes"
echo "============================================="
