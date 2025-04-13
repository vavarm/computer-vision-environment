from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import nibabel as nib
import numpy as np

from core.constants import YOLO_BOXES_FOLDER_PATH, CT_FOLDER_PATH, SEG_FOLDER_PATH, DATASET_MSKCC
from utils.colors import generate_color_from_id


def visualize_segmentation_with_boxes(file_id):
    """
    Visualizes a CT scan with overlaid segmentation masks and YOLO-format bounding boxes.

    Parameters:
    -----------
    ct_file_path : Path
        Path to the CT scan (.nii.gz) file.
    seg_file_path : Path
        Path to the segmentation (.nii.gz) file.
    label_dir : Path
        Directory containing YOLO-format label files for each slice.
    file_id : str
        The patient/study identifier used to locate label files.
    """

    # Set up file paths
    label_dir = Path(YOLO_BOXES_FOLDER_PATH)
    filename = f"{file_id}.nii.gz"
    ct_path = Path(CT_FOLDER_PATH, DATASET_MSKCC)
    ct_file_path = Path(ct_path, filename)
    seg_path = Path(SEG_FOLDER_PATH, DATASET_MSKCC)
    seg_file_path = Path(seg_path, filename)

    # Check if the CT and segmentation files exist
    if not ct_file_path.exists() or not seg_file_path.exists():
        print(f"Error: Could not find CT or segmentation files for {id}.")
        return

    # Load CT and segmentation data
    ct = nib.load(str(ct_file_path))
    seg = nib.load(str(seg_file_path))

    ct_data = ct.get_fdata()
    seg_data = seg.get_fdata()

    num_slices = seg_data.shape[2]

    # Set up the matplotlib figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    def update_slice(z):
        """
        Updates the visualization to show a specific slice.

        Parameters:
        -----------
        z : int
            The index of the slice to display.
        """
        z = int(z)
        ax.clear()
        ax.set_title(f"Slice {z}")
        ax.imshow(ct_data[:, :, z], cmap='gray')  # Display CT image for the current slice

        # Display grayscale CT image
        ax.imshow(ct_data[:, :, z], cmap='gray')

        # Extract segmentation mask for the current slice
        seg_slice = seg_data[:, :, z]

        # Create RGBA overlay for segmentation
        overlay = np.zeros((*seg_slice.shape, 4))  # RGBA format

        unique_labels = np.unique(seg_slice)
        for label_id in unique_labels:
            if label_id == 0:
                continue  # Skip background label
            mask = seg_slice == label_id
            color = generate_color_from_id(label_id)
            overlay[mask] = (*color, 0.5)  # Semi-transparent overlay

        ax.imshow(overlay)

        # Load bounding boxes if present for this slice
        label_file = label_dir / f"{file_id}-{z}.txt"
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

            # Draw bounding boxes
            for class_id, x1, y1, x2, y2 in slice_boxes:
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

    # Add slider for navigating slices
    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
    slice_slider = Slider(slider_ax, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)
    slice_slider.on_changed(update_slice)

    # Initialize display with first slice
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