from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import tkinter as tk

from core.constants import YOLO_BOXES_FOLDER_PATH, CT_FOLDER_PATH, SEG_FOLDER_PATH, DATASET_MSKCC
from utils.colors import generate_color_from_id

class CTViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CT Viewer with Segmentation and YOLO Boxes")

        # Layout
        self.left_frame = tk.Frame(master, width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # File Listbox
        self.listbox = tk.Listbox(self.left_frame, width=30, font=("Courier", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # File loading
        ct_dir = Path(CT_FOLDER_PATH) / DATASET_MSKCC
        self.nii_files = sorted(ct_dir.glob("*.nii*"))  # Match both .nii and .nii.gz files
        print(self.nii_files[0].suffix)
        self.file_ids = [f.stem.replace('.nii', '') if f.name.endswith('.nii.gz') else f.stem for f in self.nii_files]
        for fid in self.file_ids:
            self.listbox.insert(tk.END, fid)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.slider_ax = None
        self.slice_slider = None

        # Load the first file by default
        if self.file_ids:
            self.load_file(self.file_ids[0])
            self.listbox.select_set(0)

    def on_file_select(self, event):
        selected = self.listbox.curselection()
        if selected:
            file_id = self.file_ids[selected[0]]
            self.load_file(file_id)

    def load_file(self, file_id):
        self.file_id = file_id
        ct_path = Path(CT_FOLDER_PATH) / DATASET_MSKCC / f"{file_id}.nii.gz"  # Try .nii.gz first
        if not ct_path.exists():
            ct_path = Path(CT_FOLDER_PATH) / DATASET_MSKCC / f"{file_id}.nii"  # If .nii.gz is not found, try .nii

        seg_path = Path(SEG_FOLDER_PATH) / DATASET_MSKCC / f"{file_id}.nii.gz"  # Try .nii.gz first
        if not seg_path.exists():
            seg_path = Path(SEG_FOLDER_PATH) / DATASET_MSKCC / f"{file_id}.nii"  # If .nii.gz is not found, try .nii

        self.label_dir = Path(YOLO_BOXES_FOLDER_PATH)

        if not ct_path.exists() or not seg_path.exists():
            print(f"Missing files for {file_id}")
            return

        self.ct_data = nib.load(str(ct_path)).get_fdata()
        self.seg_data = nib.load(str(seg_path)).get_fdata()
        self.num_slices = self.seg_data.shape[2]

        # Redraw slider
        if self.slice_slider:
            self.slider_ax.remove()
        self.slider_ax = self.fig.add_axes((0.2, 0.05, 0.6, 0.03))
        self.slice_slider = Slider(self.slider_ax, 'Slice', 0, self.num_slices - 1, valinit=0, valstep=1)
        self.slice_slider.on_changed(self.update_slice)

        # Show first slice
        self.update_slice(0)

    def update_slice(self, z):
        z = int(z)
        self.ax.clear()
        self.ax.set_title(f"{self.file_id} - Slice {z}")
        self.ax.imshow(self.ct_data[:, :, z], cmap='gray')

        seg_slice = self.seg_data[:, :, z]
        overlay = np.zeros((*seg_slice.shape, 4))
        for label_id in np.unique(seg_slice):
            if label_id == 0:
                continue
            mask = seg_slice == label_id
            color = generate_color_from_id(label_id)
            overlay[mask] = (*color, 0.5)

        self.ax.imshow(overlay)

        label_file = self.label_dir / f"{self.file_id}-{z}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    h_img, w_img = self.ct_data.shape[:2]
                    x1 = (cx - w / 2) * w_img
                    y1 = (cy - h / 2) * h_img
                    x2 = (cx + w / 2) * w_img
                    y2 = (cy + h / 2) * h_img
                    color = generate_color_from_id(class_id+1)
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor='none')
                    self.ax.add_patch(rect)
                    self.ax.text(x1, y1 - 2, f"Class {int(class_id)}", color=color, fontsize=8)

        self.canvas.draw_idle()


if __name__ == '__main__':
    root = tk.Tk()
    app = CTViewerApp(root)
    root.mainloop()
