# MVTec AD Metal Nut Background Removal with U2Net

Automated background removal for the `metal_nut` category of the MVTec AD dataset using [U2Net](https://github.com/xuebinqin/u-2-net). It replaces the original background with a solid green color and provides a side-by-side comparison of the original and processed images.

## Features
- **Model:** U2Net architecture (patched for modern PyTorch compatibility).
- **Hardware Acceleration:** Native support for Apple Silicon (MPS), CUDA, and CPU.
- **Progress Tracking:** Interactive progress bar using `tqdm`.
- **Performance:** High-speed inference (~29ms per image on M1 Pro).
- **Visuals:** Generates side-by-side (Original | Processed) images.

## Prerequisites
- macOS (with MPS support) or Linux/Windows (with CUDA/CPU).
- [uv](https://github.com/astral-sh/uv) for package management.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/birkhofflee/u2net-mvtec-ad-metal-nuts-bg-removal
   cd u2net-mvtec-ad-metal-nuts-bg-removal
   ```

2. **Install dependencies:**
   The project uses `uv` to manage its virtual environment and dependencies.
   ```bash
   uv sync
   ```

3. **Model Weights:**
   Ensure `u2net.pth` is present in the root directory, download from [xuebinqin/u-2-net](https://github.com/xuebinqin/u-2-net).

4. Dataset location:
   Download the MVTec AD dataset from [Hugging Face](https://huggingface.co/datasets/Voxel51/mvtec-ad), and set the path in [main.py](main.py)

## Usage

Run the main inference script:
```bash
uv run python main.py
```

The script will:
1. Detect the best available hardware (MPS, CUDA, or CPU).
2. Load all 335 samples (good and abnormal) from the `metal_nut` category.
3. Process each image and save the result to the `results/` directory.
4. Display average inference speed upon completion.

## Dataset Details
- **Dataset:** MVTec AD (Anomaly Detection).
- **Category:** `metal_nut`.
- **Total Samples:** 335 (220 training, 115 test).
- **Defect Types:** good, color, flip, scratch, bent.

## Performance
- **Device:** Apple Silicon (MPS)
- **Inference Speed:** ~28.76 ms/image
- **Throughput:** ~7.28 images/second

## Main code files
- `main.py`: Inference and post-processing logic.
- `u2net.py`: U2Net model architecture.

## Acknoledgements

- Thanks to U2Net team for the model and some code.

