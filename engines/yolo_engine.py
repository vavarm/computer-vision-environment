import os
import shutil
import textwrap
from pathlib import Path
from ultralytics import YOLO
import nibabel as nib
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from core.constants import *


def create_datasets(nii_dir, labels_dir, yolo_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Converts a folder of NIfTI files and corresponding YOLO label files into a dataset formatted for YOLO training.

    Args:
        nii_dir (Path): Path to the directory containing .nii or .nii.gz files.
        labels_dir (Path): Path to the directory containing YOLO .txt annotation files.
        yolo_dir (str): Output directory where the YOLO dataset structure will be created.
        split_ratio (tuple): Tuple representing the train/val/test split ratios.
    """
    # Ensure yolo directory structure exists
    os.makedirs(f"{yolo_dir}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/images/val", exist_ok=True)
    os.makedirs(f"{yolo_dir}/images/test", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/val", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/test", exist_ok=True)

    # Get list of nii files
    nii_files = [f for f in nii_dir.rglob("*.nii*")]
    all_images = []

    nii_basename = lambda f: f.name.replace('.nii.gz', '') if f.suffix == '.gz' else f.stem

    # Loop through each nii file
    for nii_file in nii_files:
        img = nib.load(str(nii_file))
        img_data = img.get_fdata()

        # Loop through each slice
        for slice_idx in range(img_data.shape[2]):  # Iterate through z-slices
            image_slice = img_data[:, :, slice_idx]
            print(f"Slice {nii_file.name} [{slice_idx}]: shape = {image_slice.shape}")

            # Convert the image slice to an appropriate format
            slice_filename = f"{nii_basename(nii_file)}-{slice_idx}.png"
            slice_path = os.path.join(yolo_dir, "images", slice_filename)

            # Save the image slice
            slice_img = Image.fromarray(image_slice.astype(np.uint8))
            slice_img.save(slice_path)

            # Handle the label files
            label_file_path = os.path.join(labels_dir, f"{nii_basename(nii_file)}-{slice_idx}.txt")
            if os.path.exists(label_file_path):
                label_dest_path = os.path.join(yolo_dir, "labels", slice_filename.replace('.png', '.txt'))
                shutil.copy(label_file_path, label_dest_path)
            else:
                print(f"Warning: label file {label_file_path} does not exist")

            # Track the image
            all_images.append(slice_filename)

    # Split the data into train, validation, and test sets
    train_files, test_files = train_test_split(all_images, test_size=split_ratio[2], random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                                              random_state=42)

    # Move files to their respective folders
    def move_files(file_list, set_name):
        for file in file_list:
            img_file = os.path.join(yolo_dir, "images", file)
            label_file = os.path.join(yolo_dir, "labels", file.replace('.png', '.txt'))
            shutil.move(img_file, os.path.join(yolo_dir, "images", set_name, file))
            if os.path.exists(label_file):
                shutil.move(label_file, os.path.join(yolo_dir, "labels", set_name, file.replace('.png', '.txt')))

    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

    # Detect unique class IDs
    all_label_files = list(Path(yolo_dir, 'labels').rglob("*.txt"))
    class_ids = set()

    for label_path in all_label_files:
        with open(label_path, "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_ids.add(class_id)

    class_names = [f"class_{i}" for i in sorted(class_ids)]
    nc = len(class_names)

    # Create the YAML config
    yaml_content = textwrap.dedent(f"""\
        train: {os.path.abspath(os.path.join(yolo_dir, "images", "train"))}
        val: {os.path.abspath(os.path.join(yolo_dir, "images", "val"))}
        test: {os.path.abspath(os.path.join(yolo_dir, "images", "test"))}

        nc: {nc}
        names: {class_names}
        """)

    with open(os.path.join(yolo_dir, 'dataset.yaml'), 'w') as yaml_file:
        yaml_file.write(yaml_content)

    print(f"\nDataset prepared ✅\nNumber of classes: {nc}\nClasses: {class_names}\n")


# Train function placeholder
def train(dataset_path):
    print(f"Training with dataset from: {dataset_path}")

    # Load the YOLOv8n model
    model = YOLO("yolov8n.pt")  # Load the YOLOv8n model (pre-trained weights)

    # Train the model using the dataset path
    model.train(
        data=f"{dataset_path}/dataset.yaml",  # Path to dataset.yaml
        imgsz=512,  # Image size
        batch=16,  # Batch size
        epochs=40,  # Number of epochs
        device=0  # Use GPU 0 (or "cpu" if using CPU)
    )

# Main script
if __name__ == '__main__':
    _nii_dir = Path(CT_FOLDER_PATH, DATASET_MSKCC)
    _labels_dir = YOLO_BOXES_FOLDER_PATH
    _yolo_dir = 'yolo'

    create_datasets(_nii_dir, _labels_dir, _yolo_dir)

    # Training
    _dataset_path = _yolo_dir
    train(_dataset_path)