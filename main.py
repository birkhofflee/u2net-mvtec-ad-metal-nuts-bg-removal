import os
import json
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from u2net import U2NET
from tqdm import tqdm

# Configuration
DATASET_ROOT = "/Users/ale/fiftyone/huggingface/hub/Voxel51/mvtec-ad/"
SAMPLES_JSON = os.path.join(DATASET_ROOT, "samples.json")
MODEL_PATH = "u2net.pth"
OUTPUT_DIR = "results"
CATEGORY = "metal_nut"

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def remove_background(model, img_path, device):
    # Load and preprocess image
    input_image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = input_image.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Inference with timing
    start_time = time.time()
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)
    inference_time = time.time() - start_time

    # Post-process mask
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    predict_np = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('L')
    mask = im.resize((orig_w, orig_h), resample=Image.BILINEAR)

    # Apply mask
    mask_np = np.array(mask) / 255.0
    img_np = np.array(input_image)

    # Green background
    green_bg = np.zeros_like(img_np)
    green_bg[:] = [0, 255, 0] # RGB for green

    # Blend image with green background based on mask
    mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
    result_np = (img_np * mask_3d + green_bg * (1 - mask_3d)).astype(np.uint8)
    result_image = Image.fromarray(result_np)

    # Concatenate original and result
    final_image = Image.new('RGB', (orig_w * 2, orig_h))
    final_image.paste(input_image, (0, 0))
    final_image.paste(result_image, (orig_w, 0))

    return final_image, inference_time

def main():
    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    model = U2NET(3, 1)
    try:
        # Load weights on CPU first to avoid device mismatch issues before sending to MPS
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Load samples
    if not os.path.exists(SAMPLES_JSON):
        print(f"samples.json not found at {SAMPLES_JSON}")
        return

    with open(SAMPLES_JSON, 'r') as f:
        data = json.load(f)

    samples = [s for s in data['samples'] if s['category']['label'] == CATEGORY]
    print(f"Found {len(samples)} samples for {CATEGORY}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_inference_time = 0

    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        img_path = os.path.join(DATASET_ROOT, sample['filepath'])
        if not os.path.exists(img_path):
            continue

        result, inf_time = remove_background(model, img_path, device)
        total_inference_time += inf_time

        orig_filename = os.path.basename(sample['filepath'])
        output_filename = f"{CATEGORY}_{sample['split']}_{sample['defect']['label']}_{orig_filename}"
        result.save(os.path.join(OUTPUT_DIR, output_filename))

    avg_time = (total_inference_time / len(samples)) * 1000 if samples else 0
    print(f"\nProcessing complete!")
    print(f"Average Inference Speed: {avg_time:.2f} ms per image")
    print(f"Results are in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
