# File: 1_dataset_generation/generate_synthetic.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from physics_model import UnderwaterPhysicsModel

# --- Settings ---
INPUT_DIR = "input_assets"
OUTPUT_DIR = "output_dataset"
# Use 80% of images for training, 20% for testing
TRAIN_SPLIT_RATIO = 0.8
IMG_SIZE = 256
# ------------------

# Prepare image converter
transformer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor() # Converts PIL image (0-255) to Tensor (0.0-1.0)
])

to_pil = transforms.ToPILImage()

# Load our "muddifier" model
model = UnderwaterPhysicsModel()
model.eval() # Set it to evaluation mode

# Get all our clean input pictures
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    print(f"Error: No images found in '{INPUT_DIR}'. Please add images first.")
    exit()

np.random.shuffle(image_files)

# Split into train and test
split_index = int(len(image_files) * TRAIN_SPLIT_RATIO)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# Create folders
os.makedirs(os.path.join(OUTPUT_DIR, "train/clear"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train/degraded"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test/clear"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test/degraded"), exist_ok=True)

print(f"Found {len(image_files)} images. Starting generation...")

def process_files(file_list, split):
    count = 0
    for filename in file_list:
        try:
            # 1. Load the clean image
            img_path = os.path.join(INPUT_DIR, filename)
            J_pil = Image.open(img_path).convert("RGB")
            J = transformer(J_pil).unsqueeze(0) # Add batch dimension [1, 3, 256, 256]

            # 2. Pick a random "depth" to simulate fogginess
            depth = torch.tensor(np.random.uniform(5.0, 15.0)).float().view(1, 1, 1, 1)

            # 3. Create the "dirty" image
            with torch.no_grad():
                I, _, _ = model(J, depth)

            I_pil = to_pil(I.squeeze(0)) # Remove batch dimension

            # 4. Save both pictures
            # We resize the original to match the 256x256 output size
            clear_pil_resized = J_pil.resize((IMG_SIZE, IMG_SIZE))

            clear_path = os.path.join(OUTPUT_DIR, split, "clear", filename)
            degraded_path = os.path.join(OUTPUT_DIR, split, "degraded", filename)

            clear_pil_resized.save(clear_path)
            I_pil.save(degraded_path)
            count += 1
        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")
    print(f"Generated {count} images for {split} split.")

print("--- Processing Train Set ---")
process_files(train_files, "train")

print("--- Processing Test Set ---")
process_files(test_files, "test")

print("Done! Your dataset is ready in 'output_dataset/'.")