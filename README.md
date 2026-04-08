
# ComfyUI Pano2Room 

A native ComfyUI wrapper for **[Pano2Room (SIGGRAPH Asia 2024)](https://arxiv.org/abs/2408.11413)**. 

This custom node converts a single 360° equirectangular panorama into a fully explorable 3D Gaussian Splatting (.ply) environment. We have cracked open the original rigid pipeline to expose a suite of professional VFX parameters, allowing you to art-direct the room scale, patch quality, and fine-tuning models directly from the ComfyUI graph.

---

## The Dependency Hell (Read Before Installing)
The original Pano2Room research repository is incredible, but it relies on a highly fragile, outdated web of Python dependencies. If you attempt a standard `pip install -r requirements.txt` on a modern cloud GPU (like RunPod with PyTorch 2.4+ and CUDA 12+), **the 3D engine will instantly crash.** To make this usable for VFX artists, there is a custom installer script that automatically applies the following critical patches:

1. **The `xformers` Bypass:** The original code hard-requires legacy `xformers` for memory efficiency, which fails to compile on modern PyTorch versions. This script patches the source code to natively use PyTorch 2.0+ Scaled Dot Product Attention (SDPA) instead.
2. **The Hugging Face API Fix:** The original code tries to download `stable-diffusion-2-inpainting` from a gated StabilityAI repository, which throws a `401 Unauthorized` error. This script repoints the codebase to the ungated community mirror.
3. **The Nuke & Lock Protocol:** Modern versions of `numpy`, `scipy`, and `open3d` fundamentally break the 3DGS rasterizer. The script forcefully purges conflicting packages and locks them to the exact versions required for the math to work.

---

## Installation (Linux / RunPod)

**Step 1: Clone the Repository**
Navigate to your ComfyUI custom nodes folder and clone this repo:
```bash
cd /workspace/ComfyUI/custom_nodes
git clone [https://github.com/TrickyGo/Pano2Room.git](https://github.com/TrickyGo/Pano2Room.git) comfyui-pano2room

```

**Step 2: Create the Master Installer**
Navigate into the folder and create a shell script:

```bash
cd comfyui-pano2room
nano install_pano2room.sh

```

Paste the following installation script into the file and save it:

```bash
#!/bin/bash
set -e
P2R_DIR="/workspace/ComfyUI/custom_nodes/comfyui-pano2room"

echo "============================================"
echo "  Pano2Room: Standalone Installation"
echo "============================================"

# 0. SYSTEM LIBRARIES
echo "[1/6] Installing system libraries..."
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libopencv-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev libboost-all-dev libeigen3-dev libgl1-mesa-glx 2>&1 | tail -3

# 1. COMFYUI CORE DEPS
echo "[2/6] Installing ComfyUI requirements..."
cd /workspace/ComfyUI
pip install -r requirements.txt --break-system-packages -q 2>&1 | tail -3

# 2. PIP PACKAGES (Base)
echo "[3/6] Installing pip packages..."
pip install --break-system-packages --no-cache-dir -q \
    "opencv-python==4.9.0.80" \
    matplotlib timm colorama omegaconf plyfile pybind11 addict kiui \
    torchmetrics scikit-image plotly \
    2>&1 | tail -3

# 3. CUDA SUBMODULES (Rasterizer & KNN)
echo "[4/6] Compiling CUDA submodules locally..."
cd "$P2R_DIR"
mkdir -p submodules
cd submodules

if [ -d "diff-gaussian-rasterization" ]; then
    rm -rf diff-gaussian-rasterization
fi
git clone --recursive [https://github.com/graphdeco-inria/diff-gaussian-rasterization.git](https://github.com/graphdeco-inria/diff-gaussian-rasterization.git)
cd diff-gaussian-rasterization
pip install . --break-system-packages --no-build-isolation -q 2>&1 | tail -3
cd ..

if [ -d "simple-knn" ]; then
    rm -rf simple-knn
fi
git clone [https://github.com/camenduru/simple-knn.git](https://github.com/camenduru/simple-knn.git)
cd simple-knn
pip install . --break-system-packages --no-build-isolation -q 2>&1 | tail -3
cd ../..

# 4. COMFYUI NODE DEPS (The Version Lock Merge)
echo "[5/6] Restoring Node Dependencies & Version Synchronization..."
pip install --break-system-packages -q \
    addict rich GitPython soundfile replicate requests toml packaging h5py \
    "datasets==2.18.0" webdataset clip trimesh kornia omegaconf easydict pytorch-lightning \
    diffusers icecream tifffile imageio imageio-ffmpeg accelerate "peft==0.10.0" \
    "albumentations==1.1.0" \
    "imgaug==0.4.0" \
    "huggingface_hub==0.22.2" \
    "transformers==4.39.3" \
    2>&1 | tail -3

# 5. THE "NUKE & LOCK" PROTOCOL
echo "[6/6] Executing Engine Lockdown..."
pip install --break-system-packages --no-cache-dir --ignore-installed -q blinker
pip uninstall -y numpy scipy open3d >/dev/null 2>&1 || true
rm -rf /usr/local/lib/python3.11/dist-packages/numpy*
rm -rf /usr/local/lib/python3.11/dist-packages/scipy*
rm -rf /usr/local/lib/python3.11/dist-packages/open3d*

pip install --break-system-packages --no-cache-dir -q \
    "open3d==0.18.0" \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    2>&1 | tail -3

# --- THE PANO2ROOM SOURCE CODE PATCHES ---
echo "  Patching Pano2Room HF Repo to ungated community version..."
sed -i 's/stabilityai\/stable-diffusion-2-inpainting/sd2-community\/stable-diffusion-2-inpainting/g' /workspace/ComfyUI/custom_nodes/comfyui-pano2room/modules/inpainters/PanoPersFusionInpainter.py
sed -i 's/stabilityai\/stable-diffusion-2-inpainting/sd2-community\/stable-diffusion-2-inpainting/g' /workspace/ComfyUI/custom_nodes/comfyui-pano2room/train_SDFT.py

echo "  Bypassing legacy xformers for native PyTorch 2.0 SDPA..."
sed -i 's/args.enable_xformers_memory_efficient_attention/False/g' /workspace/ComfyUI/custom_nodes/comfyui-pano2room/train_SDFT.py

echo "============================================"
echo "  Standalone Installation Complete!"
echo "============================================"

```

**Step 3: Sanitize and Run**
If you copy-pasted the script in Windows, invisible carriage returns (`\r`) will break the execution in Linux. Run this command to sanitize the file, then execute it:

```bash
sed -i 's/\r$//' install_pano2room.sh
sh install_pano2room.sh

```

Restart ComfyUI, and you are ready to go!

---

## Included Nodes

### 1. Pano2Room (Panorama → 3DGS)

The core generation engine. Takes an equirectangular image and outputs a `3DGS.ply` file, a mesh, and novel view renders.

**VFX Parameters Exposed:**

* `run_sdft`: Toggle to fine-tune a custom LoRA on your room before generating the 3D space. Highly recommended for accurate lighting.
* `sdft_training_steps`: How long the AI trains on your room's lighting (Default: 500). Higher = more accurate textures, but risks overfitting.
* `inpaint_stride`: **(Crucial for Quality)** Controls how often the camera stops to patch holes. `20` is fast but misses narrow gaps. `5` is highly meticulous and eliminates most "black holes" (takes longer).
* `room_scale`: Physically dilate or constrict the scale of the room geometry. (<1.0 = larger space, >1.0 = smaller space).
* `render_resolution`: Dial in the internal working resolution (512, 768, 1024) for sharper wood grains and fabric textures.
* `save_debug_passes`: Dumps raw alpha masks and depth passes into the output folder for external compositing in Nuke or After Effects.

### 2. Pano2Room Camera Trajectory

Generates mathematical 4x4 `[R|T]` extrinsic matrices for rendering fly-throughs of your new 3D room.

* **Modes:** `orbit`, `dolly_forward`, `dolly_lateral`, `custom_circle`.
* Controllable radius, height offsets, and frame counts.

### 3. Pano2Room PLY Loader

A simple utility node to pass the generated `.ply` file path to external 3DGS viewer nodes.

---

## Workflow Example

1. Load a 360° panorama image using `Load Image`.
2. Connect it to the **Pano2Room (Panorama → 3DGS)** node.
3. Set `inpaint_stride` to `5` for a high-quality, hole-free generation.
4. Connect the output `trajectory_dir` from a **Pano2Room Camera Trajectory** node into your final Gaussian Splat rendering node.
5. Hit **Queue Prompt**.

---

## Credits & Citation

This ComfyUI wrapper is built on top of the incredible research by the Pano2Room team. If you use this in academic work, please cite the original paper:

```bibtex
Guo Pu, Yiming Zhao, and Zhouhui Lian. 2024. Pano2Room: Novel View Synthesis from a Single Indoor Panorama. In SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 3--6, 2024, Tokyo, Japan. ACM, New York, NY, USA, 10 pages.
[https://doi.org/10.1145/3680528.3687616](https://doi.org/10.1145/3680528.3687616)

```




```
