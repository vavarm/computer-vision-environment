from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import nibabel as nib
import numpy as np

from core.constants import YOLO_BOXES_FOLDER_PATH, CT_FOLDER_PATH, SEG_FOLDER_PATH, DATASET_MSKCC
from utils.colors import generate_color_from_id

def visualize_segmentation_with_boxes(ct_file_path, seg_file_path, label_dir, id):

    # Check if files exist
    if not ct_file_path.exists() or not seg_file_path.exists():
        print(f"Error: Could not find CT or segmentation files for {id}.")
        return

    # Load CT and segmentation data
    ct = nib.load(str(ct_file_path))
    seg = nib.load(str(seg_file_path))

    ct_data = ct.get_fdata()
    seg_data = seg.get_fdata()

    num_slices = seg_data.shape[2]

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

        label_file = label_dir / f"{id}-{z}.txt"
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

if __name__ == '__main__':
    _basename = "330680"

    _label_dir = Path(YOLO_BOXES_FOLDER_PATH)
    _filename = f"{_basename}.nii.gz"
    _ct_path = Path(CT_FOLDER_PATH, DATASET_MSKCC)
    _ct_file_path = Path(_ct_path, _filename)
    _seg_path = Path(SEG_FOLDER_PATH, DATASET_MSKCC)
    _seg_file_path = Path(_seg_path, _filename)
    # Visualize segmentation with boxes
    visualize_segmentation_with_boxes(_ct_file_path, _seg_file_path, _label_dir, _basename)