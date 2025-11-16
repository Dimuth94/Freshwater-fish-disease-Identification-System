import os
import cv2
import albumentations as A
from tqdm import tqdm

# PARAMETERS
SRC_DIR = "D:\MSC\Research\Freshwater-fish-disease-Identification-System\freshwater_fish_disease\data\dataset\train"
DEST_DIR = "D:\MSC\Research\Freshwater-fish-disease-Identification-System\freshwater_fish_disease\data\augmented_dataset\train"
N = 2  # Number of augmented images per original

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.Blur(blur_limit=3, p=0.2),
])

# Create output folders
os.makedirs(DEST_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    class_path = os.path.join(SRC_DIR, class_name)
    save_path = os.path.join(DEST_DIR, class_name)
    os.makedirs(save_path, exist_ok=True)

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Skipping {img_path} (cannot read)")
            continue

        # Save original
        cv2.imwrite(os.path.join(save_path, img_name), img)

        # Generate N augmentations
        base_name, ext = os.path.splitext(img_name)
        for i in range(1, N + 1):
            augmented = transform(image=img)["image"]
            aug_name = f"{base_name}_aug{i}{ext}"
            cv2.imwrite(os.path.join(save_path, aug_name), augmented)
