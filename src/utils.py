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
from PIL import Image
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import QApplication, QProgressDialog, QMessageBox

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
    TARGET_RES = 0.022724609375
    tif = tifffile.TiffFile(tif_fl)
    if "XResolution" in tif.pages[0].tags:
        x1, x2 = tif.pages[0].tags["XResolution"].value
        if tif.pages[0].tags["ResolutionUnit"].value == 3:  # RESUNIT.CENTIMETER
            x2 = x2 * 10000
        return (x2 / x1) * (tif.pages[0].shape[0] / sh)
    else:
        return TARGET_RES


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


def fill_with_colors(_image, _mask, ncomp, cols):
    for nc in range(1, ncomp + 1):
        rgb = list(colorsys.hsv_to_rgb(cols[nc - 1], 0.75, 1))
        rgb.append(0.5)
        xs, ys = np.where(_mask == nc)

        for i in range(xs.size):
            _image[xs[i], ys[i], :] = rgb
    return _image


def plot_labels(_image, _instance_mask, _semantic_mask, ncomp, _output_file):
    instance_layer = np.zeros((*_instance_mask.shape, 4))
    semantic_layer = np.zeros((*_instance_mask.shape, 4))
    col_inds = [float(nc) / ncomp for nc in range(ncomp)]
    col_sem = [float(nc) / 3 for nc in range(3)]
    random.shuffle(col_inds)

    instance_layer = fill_with_colors(instance_layer, _instance_mask, ncomp, col_inds)
    semantic_layer = fill_with_colors(semantic_layer, _semantic_mask, 3, col_sem)
    instance_layer = np.append(instance_layer, semantic_layer, axis=1)
    background = np.append(_image[0, :, :], _image[0, :, :], axis=1)

    instance_layer = Image.fromarray((instance_layer * 255).astype(np.uint8), mode="RGBA")
    background = Image.fromarray(background * 255).convert("RGBA")

    Image.alpha_composite(background, instance_layer).save(_output_file)


def open_dir_in_browser(_path):
    if sys.platform == 'win32':
        os.startfile(_path)
    elif sys.platform == 'darwin':
        subprocess.call(['open', _path])
    else:
        subprocess.call(['xdg-open', _path])
