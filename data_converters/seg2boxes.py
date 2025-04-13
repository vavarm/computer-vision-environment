import os
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label

from core.constants import SEG_FOLDER_PATH, DATASET_MSKCC, YOLO_BOXES_FOLDER_PATH

def generate_yolo_box(region, shape):
    minr, minc, maxr, maxc = region.bbox
    center_x = (minc + maxc) / 2 / shape[1]
    center_y = (minr + maxr) / 2 / shape[0]
    width = (maxc - minc) / shape[1]
    height = (maxr - minr) / shape[0]
    return center_x, center_y, width, height


def process_segmentation_file(seg_path, output_root):
    seg_path = Path(seg_path)
    seg = nib.load(str(seg_path))
    seg_data = seg.get_fdata()
    image_shape = seg_data.shape[:2]

    basename = seg_path.stem.replace('.nii', '')

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for z in range(seg_data.shape[2]):
        label_file = Path(output_root, f"{basename}-{z}.txt")
        with open(label_file, 'w') as f:
            slice_data = seg_data[:, :, z]
            if np.any(slice_data):
                labeled = label(slice_data)
                props = regionprops(labeled)
                for region in props:
                    center_x, center_y, width, height = generate_yolo_box(region, image_shape)
                    class_id = int(slice_data[tuple(region.coords[0])])  # Use the value of one voxel as class
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        print(f"Generated labels for slice {z} -> {label_file}")

def process_segmentation_folder(folder, output_root):
    seg_folder_path = Path(folder)

    if not seg_folder_path.exists():
        print(f"Folder {seg_folder_path} does not exist")
        return

    for seg_path in seg_folder_path.iterdir():
        process_segmentation_file(seg_path, output_root)


if __name__ == "__main__":

    # Generate YOLO box files for one segmentation
    #seg_path = Path(SEG_BASE_PATH, DATASET, f"{FILENAME}.nii.gz")
    #process_segmentation_file(seg_path)

    # Generate YOLO box files for all segmentations of a class
    _seg_folder_path = Path(SEG_FOLDER_PATH, DATASET_MSKCC)
    process_segmentation_folder(_seg_folder_path, YOLO_BOXES_FOLDER_PATH)
