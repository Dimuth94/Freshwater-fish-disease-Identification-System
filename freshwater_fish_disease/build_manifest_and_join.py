#!/usr/bin/env python3
"""
Build an images_manifest.csv by scanning your dataset folders and join to treatments.
Adjust FOLDER_TO_DISEASE mapping or provide disease_map.csv to customize names.
"""
import os, csv, sys
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset")
MAP_CSV      = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("disease_map.csv")
TREAT_CSV    = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("treatments.csv")

# Load map and treatments
df_map = pd.read_csv(MAP_CSV)
df_tr  = pd.read_csv(TREAT_CSV)

# Build lookup
folder2id = dict(zip(df_map.folder_name, df_map.disease_id))

rows = []
for split in ["train","val","test"]:
    split_dir = DATASET_ROOT / split
    if not split_dir.exists():
        continue
    for folder in sorted(os.listdir(split_dir)):
        class_dir = split_dir / folder
        if not class_dir.is_dir(): 
            continue
        disease_id = folder2id.get(folder)
        if disease_id is None:
            print(f"[WARN] No disease_id mapping for folder: {folder} (split={split})")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                rows.append({
                    "image_path": str(class_dir / fname),
                    "folder_name": folder,
                    "disease_id": disease_id
                })

df_imgs = pd.DataFrame(rows)

# Join to treatments to know what's available for each image's disease
df_join = df_imgs.merge(df_tr, on="disease_id", how="left")

# Save outputs
df_imgs.to_csv("images_manifest.csv", index=False)
df_join.to_csv("images_with_treatments.csv", index=False)

print("Wrote images_manifest.csv and images_with_treatments.csv")
print(df_join.head().to_string(index=False))
