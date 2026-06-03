# Python Imports
import colorsys
import os
import random
import re
import subprocess
import sys

# Library Imports
import numpy as np
import tifffile
import cv2
from PIL import Image, ImageDraw
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import QApplication, QProgressDialog, QMessageBox
from skimage.morphology import skeletonize

# Local Imports
from src.configs import PROJECT_DIR


# Local Imports


def filter_files(_regex, _dir):
    if _dir is None or _dir == '':
        return []
    files = os.listdir(_dir)
    files = list(filter(lambda x: re.match(_regex, x), files))
    return files


def filter_tiff_files(_dir):
    return filter_files(r'(.+).(tiff|tif)', _dir)


def create_projects_dir():
    if not os.path.exists(PROJECT_DIR):
        os.mkdir(f"./{PROJECT_DIR}")


# The resource sliders are 0-4 "allocation level" sliders. These helpers are the
# single source of truth for translating a level into the actual values used by
# the engine (thread count, batch size) and shown in the UI labels, so the two
# can never drift apart.
def cpu_threads_from_level(_level, _cores):
    # Level 0..4 maps to 20%..100% of the available cores. Clamp to [1, cores]
    # so legacy configs (e.g. cpu_allocation == 5) cannot exceed the core count.
    return max(1, min(_cores, round((_level + 1) / 5 * _cores)))


def cpu_percent_from_level(_level):
    return round((_level + 1) / 5 * 100)


def batch_size_from_level(_level, _base=8):
    return _base * (_level + 1)


def suggested_workers(_cpu_level, _mem_level, _max_workers):
    # Derive a sensible DataLoader worker count from the CPU and Mem levels,
    # scaled into [0, _max_workers]. More workers means more prefetch RAM, so
    # the memory level acts as a ceiling.
    suggested = round((_cpu_level + 1) / 5 * _max_workers)
    mem_ceiling = round((_mem_level + 1) / 5 * _max_workers)
    return max(0, min(suggested, mem_ceiling, _max_workers))


def mkdirs(_target_directory, _path):
    t = _path.split("/")
    for i, sub_dir in enumerate(t[:-1]):
        _target_directory = os.path.join(_target_directory, sub_dir)
        if not os.path.exists(_target_directory):
            os.mkdir(_target_directory)
    return _target_directory, t[-1]


def get_cuda_version():
    return execute_shell_command("nvcc --version | sed -n 's/^.*release \\([0-9]\\+\\.[0-9]\\+\\).*$/\\1/p'")


def execute_shell_command(_command):
    process = os.popen(_command)
    result = process.read().strip()
    process.close()
    return result


def get_resolution(tif_fl, sh):
    # Same computation as get_tiff_resolution, using the default target
    # resolution as the fallback when the TIFF has no XResolution tag.
    return get_tiff_resolution(tif_fl, sh, 0.022724609375)


def get_tiff_resolution(_tiff_file, _shape, _target_resolution):
    tif = tifffile.TiffFile(_tiff_file)
    if "XResolution" in tif.pages[0].tags:
        x1, x2 = tif.pages[0].tags["XResolution"].value
        if tif.pages[0].tags["ResolutionUnit"].value == 3:  # RESUNIT.CENTIMETER
            x2 = x2 * 10000
        return (x2 / x1) * (tif.pages[0].shape[0] / _shape)
    else:
        return _target_resolution


def create_dark_palette():
    # Use a palette to switch to dark colors:
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))
    return dark_palette


def create_message_box(_text, _icon):
    msgBox = QMessageBox()
    msgBox.setWindowIcon(_icon)
    msgBox.setWindowTitle("AMAP")
    msgBox.setFont(QFont("Times", 14))
    msgBox.setText(_text)
    return msgBox


def create_progress_dialog(_text, _title, _parent):
    progressDialog = QProgressDialog(
        _text, "", 0, 100, parent=_parent)

    progressDialog.setFont(QFont("Times", 14))
    progressDialog.setMinimum(0)
    progressDialog.setRange(0, 100)
    progressDialog.setWindowTitle(_title)

    progressDialog.setCancelButton(None)
    progressDialog.setWindowFlags(
        QtCore.Qt.WindowTitleHint | QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)

    return progressDialog


# This function is called before creating a new project, it checks:
# 1. Whether all the images have the same dimensionality or not
# 2. Whether the images are stacked or not
# 3. Reports the dimensionality of the images
def analyze_tiff_files(_path):
    is_valid = True
    is_stacked = False
    dimensions = ""
    previous_tiff_shape = None
    previous_tiff_rank = None
    not_a_tiff_file = []
    tiff_files = filter_tiff_files(_path)

    for tiff in tiff_files:
        # Prevent UI freeze
        QApplication.processEvents()

        full_file_path = os.path.join(_path, tiff)

        # noinspection PyBroadException
        try:
            tiff_image = tifffile.imread(full_file_path)
            dimensions += f"Rank: {len(tiff_image.shape)}, Shape: {tiff_image.shape}, Filename: {tiff}\n"

            if previous_tiff_shape is None:
                previous_tiff_shape = tiff_image.shape

            if previous_tiff_rank is None:
                previous_tiff_rank = len(tiff_image.shape)

            tiff_rank = len(tiff_image.shape)

            if previous_tiff_rank != tiff_rank:
                is_valid = False
                break

            if previous_tiff_shape != tiff_image.shape:
                if previous_tiff_rank == 3:
                    if not previous_tiff_shape[0] == tiff_image.shape[0]:
                        is_valid = False
                        break
                if previous_tiff_rank == 4:
                    if not previous_tiff_shape[1] == tiff_image.shape[1]:
                        is_valid = False
                        break

            # Each tiff file contains multiple images, each image can have multiple channels
            # Based on the reviewed data samples, if number of channels is 2, meaning "tiff_image.shape[1][1] == 2"
            # Only the first channel is usable, and the images are not stacked.
            # If no of channels is bigger than 2, the tiff image contains multiple stacked images
            if tiff_rank == 4 and (tiff_image.shape[1] > 2 or tiff_image.shape[0] > 2):
                is_stacked = True
            elif tiff_rank == 3 and tiff_image.shape[0] > 2:
                is_stacked = True
            else:
                is_stacked = False

        except Exception:
            not_a_tiff_file.append(tiff)

    return is_valid, is_stacked, dimensions


def classify_project_inputs(_source_directory):
    """Inspect a project's TIFF inputs (reading only their shapes) and decide how
    AMAP should reduce each image to a single 2D plane.

    Returns a tuple (input_class, n_channels, description):
      * input_class: one of 'homogeneous_2d', 'homogeneous_3d',
        'homogeneous_4d' or 'heterogeneous'. read_file() in the dataset branches
        on this value.
      * n_channels: number of selectable channels for 3D/4D inputs, else None.
        Used to bound the target-channel spin box.
      * description: a short human-readable summary for the UI status label.

    The channel-axis convention matches read_file(): for 3D it is axis 0, and for
    4D it is axis 0 when its length is 2, otherwise axis 1.
    """
    shapes = []
    for file_name in filter_tiff_files(_source_directory):
        try:
            with tifffile.TiffFile(os.path.join(_source_directory, file_name)) as tif:
                shapes.append(tuple(tif.series[0].shape))
        except Exception:
            # Unreadable file: skip it for classification purposes.
            continue

    count = len(shapes)
    if count == 0:
        return 'heterogeneous', None, "No readable TIFF inputs found."

    def channel_axis_size(_shape):
        rank = len(_shape)
        if rank == 3:
            return _shape[0]
        if rank == 4:
            return _shape[0] if _shape[0] == 2 else _shape[1]
        return None

    ranks = {len(s) for s in shapes}
    if len(ranks) != 1:
        return 'heterogeneous', None, (
            "%d image(s) with mixed dimensionality — processed automatically "
            "(max projection)." % count)

    rank = ranks.pop()
    if rank == 2:
        return 'homogeneous_2d', None, (
            "%d image(s) · 2-D — used as-is (no projection or channel "
            "selection)." % count)

    if rank in (3, 4):
        channel_counts = {channel_axis_size(s) for s in shapes}
        if len(channel_counts) != 1:
            return 'heterogeneous', None, (
                "%d image(s) with inconsistent channel counts — processed "
                "automatically (max projection)." % count)
        n_channels = channel_counts.pop()
        if rank == 3:
            return 'homogeneous_3d', n_channels, (
                "%d image(s) · 3-D, %d channel(s)/slice(s) — check "
                "‘Stacked’ for a max projection, or pick a target channel."
                % (count, n_channels))
        return 'homogeneous_4d', n_channels, (
            "%d image(s) · 4-D multi-channel stack, %d channel(s) — max "
            "projection then the selected target channel." % (count, n_channels))

    return 'heterogeneous', None, (
        "%d image(s) · %d-D — processed automatically." % (count, rank))


def fill_with_colors(_image, _mask, ncomp, cols):
    for nc in range(1, ncomp + 1):
        rgb = list(colorsys.hsv_to_rgb(cols[nc - 1], 0.75, 1))
        rgb.append(0.5)
        xs, ys = np.where(_mask == nc)
        # Colour all pixels of this component at once; rgb broadcasts across
        # the (N, 4) selection. Equivalent to the old per-pixel loop.
        _image[xs, ys, :] = rgb
    return _image


def plot_labels(_image,
                _instance_mask,
                _semantic_mask,
                _roi_contours,
                _ncomp,
                _output_file):

    background = Image.fromarray(_image[0, :, :] * 255).convert("RGBA")

    instance_layer = np.zeros((*_instance_mask.shape, 4))
    semantic_layer = np.zeros((*_instance_mask.shape, 4))

    col_inds = [float(nc) / _ncomp for nc in range(_ncomp)]
    col_sem = [float(nc) / 3 for nc in range(3)]
    random.shuffle(col_inds)

    instance_layer = fill_with_colors(instance_layer, _instance_mask, _ncomp, col_inds)
    instance_image = Image.fromarray((instance_layer * 255).astype(np.uint8), mode="RGBA")
    instance_image = Image.alpha_composite(background, instance_image)
    instance_image.save(f"{_output_file}_instance.png")

    semantic_layer = fill_with_colors(semantic_layer, _semantic_mask, 3, col_sem)
    semantic_image = Image.fromarray((semantic_layer * 255).astype(np.uint8), mode="RGBA")
    semantic_image = Image.alpha_composite(background, semantic_image)
    semantic_image.save(f"{_output_file}_semantic.png")

    _semantic_mask[_semantic_mask == 1] = 0
    roi_layer = np.zeros((*_semantic_mask.shape, 4))
    col_sem = [float(nc) / 3 for nc in range(3)]
    roi_layer = fill_with_colors(roi_layer, _semantic_mask, 3, col_sem)
    roi_image = Image.fromarray((roi_layer * 255).astype(np.uint8), mode="RGBA")

    roi_image = Image.alpha_composite(background, roi_image)
    draw = ImageDraw.Draw(roi_image)
    for contour in _roi_contours:
        points = np.squeeze(contour)
        if points.shape[0] > 2:
            draw.line(tuple(map(tuple, points)), fill="red", width=1)

    roi_image.save(f"{_output_file}_roi.png")

    width = roi_image.width
    pred_image = Image.new("RGBA", (width * 3, roi_image.height))
    pred_image.paste(instance_image, (0, 0))
    pred_image.paste(semantic_image, (width, 0))
    pred_image.paste(roi_image, (width*2, 0))
    pred_image.save(f"{_output_file}_pred.png")


def get_ROI_AMAP(predictions, img_sh):
    MIN_AREA = 500

    predictions = cv2.resize(predictions, img_sh, interpolation=cv2.INTER_NEAREST)
    all_pred = predictions > 0
    all_pred = all_pred.astype(np.uint8)

    contours, _ = cv2.findContours(all_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conv_cnt = [cv2.convexHull(cont) for cont in contours]
    # filter small areas
    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > MIN_AREA, conv_cnt))

    mask_roi = np.zeros(img_sh, np.uint8)
    for i in range(len(contours)):
        mask1 = cv2.drawContours(np.zeros(img_sh, np.uint8), contours, i, 1, -1)
        mask_roi[mask1 == 1] = 1

    mask_orig = mask_roi.copy()
    kernel = np.zeros((11, 11), np.uint8)
    kernel = cv2.circle(kernel, (5, 5), 5, 1, 0)

    mask_roi = cv2.dilate(mask_roi, kernel, iterations=15)
    mask_roi = cv2.erode(mask_roi, kernel, iterations=10)
    mask_roi[mask_orig == 1] = 1

    sd = predictions.copy()
    sd[sd == 1] = 0
    sd[sd == 2] = 1
    sd[mask_roi == 0] = 0
    sd = skeletonize(sd)
    sd = sd.astype(np.uint8)

    return mask_roi, sd


def get_ROI_AMAPAPP(predictions, img_sh, dilation_iters=25, erosion_iters=8, min_area=5000):
    # Morphology iteration counts and the minimum ROI area were chosen empirically in the AMAP
    # study and replicated here; exposed as user-tweakable advanced parameters whose defaults
    # preserve the validated behaviour.
    MIN_AREA = min_area
    predictions = cv2.resize(predictions,
                             img_sh,
                             interpolation=cv2.INTER_NEAREST)

    ero_kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    dil_kernel = cv2.circle(kernel, (5, 5), 5, 1, 0)

    mask_roi = predictions.copy().astype(np.uint8)
    mask_roi[mask_roi == 1] = 0
    mask_roi = cv2.dilate(mask_roi,
                          dil_kernel,
                          iterations=dilation_iters)
    mask_roi = cv2.erode(mask_roi,
                         ero_kernel,
                         iterations=erosion_iters)

    # Find connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
    # Filter out small components
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    new_mask = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= MIN_AREA:
            new_mask[output == i + 1] = 1

    mask_roi = new_mask

    sd = predictions.copy()
    sd[sd == 1] = 0
    sd[sd == 2] = 1
    sd[mask_roi == 0] = 0
    sd = skeletonize(sd)
    sd = sd.astype(np.uint8)

    return mask_roi, sd


def get_ROI_from_predictions(predictions, img_sh, is_old,
                             dilation_iters=25, erosion_iters=8, min_area=5000):
    if is_old:
        return get_ROI_AMAP(predictions, img_sh)
    else:
        return get_ROI_AMAPAPP(predictions, img_sh,
                               dilation_iters=dilation_iters,
                               erosion_iters=erosion_iters,
                               min_area=min_area)


def open_dir_in_browser(_path):
    _path = os.path.normpath(_path)
    if sys.platform == 'win32':
        os.startfile(_path)
    elif sys.platform == 'darwin':
        subprocess.call(['open', _path])
    else:
        subprocess.call(['xdg-open', _path])

