import hashlib
import os
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

DATASET = "MSKCC"
FILENAME = "330680"

# File path placeholder
CT_BASE_PATH = "data/CT"
SEG_BASE_PATH = "data/Segmentation"
OUTPUT_ROOT = "data/yolo"


def generate_color_from_id(id):
    # A predefined set of colors based on the ID (just a few examples)
    color_map = {
        1: (0.1, 0.2, 0.5),  # Blue
        2: (0.7, 0.2, 0.1),  # Red
        3: (0.1, 0.7, 0.1),  # Green
        4: (0.9, 0.7, 0.1),  # Yellow
        5: (0.4, 0.4, 0.8),  # Light Blue
        6: (0.9, 0.3, 0.7),  # Pink
        7: (0.2, 0.9, 0.4),  # Light Green
        8: (0.8, 0.8, 0.8)  # Gray
    }

    # Return a color based on the ID, defaulting to a gray if the ID isn't in the map
    return color_map.get(id, (0.5, 0.5, 0.5))  # Default to gray if ID not found

def generate_yolo_box(region, shape):
    minr, minc, maxr, maxc = region.bbox
    center_x = (minc + maxc) / 2 / shape[1]
    center_y = (minr + maxr) / 2 / shape[0]
    width = (maxc - minc) / shape[1]
    height = (maxr - minr) / shape[0]
    return center_x, center_y, width, height


def process_segmentation_file(seg_path, output_root=OUTPUT_ROOT):
    seg_path = Path(seg_path)
    seg = nib.load(str(seg_path))
    seg_data = seg.get_fdata()
    image_shape = seg_data.shape[:2]

    # Create output directories if they don't exist
    for subfolder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(Path(output_root) / subfolder, exist_ok=True)

    basename = seg_path.stem.replace('.nii', '')
    label_dir = Path(output_root) / 'labels/train'

    for z in range(seg_data.shape[2]):
        label_file = label_dir / f"{basename}-{z}.txt"
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


def visualize_segmentation_with_boxes(dataset, filename, output_root=OUTPUT_ROOT):
    ct_path = Path(CT_BASE_PATH) / dataset / f"{filename}.nii.gz"
    seg_path = Path(SEG_BASE_PATH) / dataset / f"{filename}.nii.gz"

    # Check if files exist
    if not ct_path.exists() or not seg_path.exists():
        print(f"Error: Could not find CT or segmentation files for {filename}.")
        return

    # Load CT and segmentation data
    ct = nib.load(str(ct_path))
    seg = nib.load(str(seg_path))

    ct_data = ct.get_fdata()
    seg_data = seg.get_fdata()

    num_slices = seg_data.shape[2]
    basename = Path(seg_path).stem.replace('.nii', '')

    label_dir = Path(output_root) / 'labels/train'

    # Set up the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    def update_slice(z):
        z = int(z)
        ax.clear()
        ax.set_title(f"Slice {z}")
        ax.imshow(ct_data[:, :, z], cmap='gray')  # Display CT image for the current slice

        # Get current slice
        seg_slice = seg_data[:, :, z]

        # Create an RGB overlay image
        overlay = np.zeros((*seg_slice.shape, 4))  # RGBA

        unique_labels = np.unique(seg_slice)
        for label_id in unique_labels:
            if label_id == 0:
                continue  # skip background
            mask = seg_slice == label_id
            color = generate_color_from_id(label_id)
            overlay[mask] = (*color, 0.5)  # RGB + alpha

        ax.imshow(overlay)

        label_file = label_dir / f"{basename}-{z}.txt"
        if label_file.exists():
            slice_boxes = []
            with open(label_file, 'r') as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    h_img, w_img = ct_data.shape[:2]
                    x1 = (cx - w / 2) * w_img
                    y1 = (cy - h / 2) * h_img
                    x2 = (cx + w / 2) * w_img
                    y2 = (cy + h / 2) * h_img
                    slice_boxes.append((class_id, x1, y1, x2, y2))

            # Add boxes to the image
            for box in slice_boxes:
                class_id, x1, y1, x2, y2 = box
                color = generate_color_from_id(class_id)
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, f"Class {int(class_id)}", color=color, fontsize=8)

        fig.canvas.draw_idle()

    # Slider for slice navigation
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    slice_slider = Slider(slider_ax, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)
    slice_slider.on_changed(update_slice)

    update_slice(0)
    plt.show()


if __name__ == "__main__":

    # Generate YOLO box files
    seg_path = Path(SEG_BASE_PATH) / DATASET / f"{FILENAME}.nii.gz"
    process_segmentation_file(seg_path)

    # Visualize segmentation with boxes
    visualize_segmentation_with_boxes(DATASET, FILENAME)
