# Python Imports
import ctypes
import json
import logging
import os
import re
import math
import glob

# Library Imports
import cv2
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

# Local Imports
from src.configs import LOG_START_PROC_SIGNATURE
from src.utils import get_resolution, get_ROI_from_predictions


class AMAPMorphometry:
    def __init__(self, _configs):

        self.END_POINT = 2
        self.JUNCTION_POINT = 3
        self.SLAB_POINT = 4

        # Configuration
        self.configs = _configs
        self.project_id = _configs['project_id']
        self.project_name = _configs['project_name']
        self.base_directory = _configs['base_dir']
        self.source_directory = _configs['source_dir']
        self.npy_directory = _configs['npy_dir']
        self.output_segmentation_directory = _configs['result_segmentation_dir']
        self.output_morphometry_directory = _configs['result_morphometry_dir']

        if not os.path.exists(self.output_morphometry_directory):
            os.mkdir(self.output_morphometry_directory)

        self.no_of_processed_images = mp.Array(ctypes.c_int64, 1)
        self.no_of_images = mp.Array(ctypes.c_int64, 1)

        self.no_of_images.acquire()
        self.no_of_images[0] = 100
        self.no_of_images.release()

    def exec(self):
        logging.info(LOG_START_PROC_SIGNATURE)
        logging.info("Morphometry process started")

        self.foot_processes_parameter_table(self.source_directory,
                                            self.npy_directory,
                                            self.output_morphometry_directory)

        self.combine_FP_SD(self.output_morphometry_directory)
        logging.info("Morphometry process finished")

        self.configs['is_morphometry_finished'] = True
        config_file_path = os.path.join(self.base_directory, "conf.json")
        with open(config_file_path, 'w+') as file:
            file.write(json.dumps(self.configs))

    def skeleton_length(self, input_image, res):
        tagged_image = self.tag_image(input_image)
        return self.mark_trees(tagged_image, res)

    def foot_processes_parameter_table(self, images_dir, prediction_dir, output_dir):
        files = list(filter(lambda entry: re.match(r'(.+)_pred.npy', entry), os.listdir(prediction_dir)))
        filenames = [re.match(r'(.+)_pred.npy', x).group(1) for x in files]
        filenames.sort()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.no_of_images.acquire()
        self.no_of_images[0] = len(filenames)
        self.no_of_images.release()

        if self.configs['does_include_sd']:
            sd_grid_file = open(os.path.join(output_dir, "SD_length_grid_index.csv"), 'w')
        try:

            if self.configs['does_include_sd']:
                sd_grid_file.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (
                    "file", "SD length", "grid crossings", "mean distance", "SD total length", "ROI total area"))

            for i, filename in enumerate(filenames):

                logging.info(f"Saving file: {filename}")

                self.no_of_processed_images.acquire()
                self.no_of_processed_images[0] = i
                self.no_of_processed_images.release()

                predictions = np.load(os.path.join(prediction_dir, filename + "_pred.npy"))

                instance_prediction = predictions[0, :, :]
                values = np.unique(instance_prediction)
                values = values[values != 0]
                resolution = get_resolution(os.path.join(images_dir, filename + ".tif"), predictions.shape[1])
                with open(os.path.join(output_dir, filename + "_fp_params.csv"), 'w') as csv_file:
                    csv_file.write("Label\tArea\tPerim.\tCirc.\n")
                    for value in values:
                        is_value = instance_prediction == value
                        per, area, circ = self.foot_process_parameters(is_value, resolution)
                        csv_file.write("%i\t%.3f\t%.3f\t%.3f\n" % (value, area, per, circ))
                    csv_file.close()

                sd = predictions[1, :, :]
                roi_mask, sd = get_ROI_from_predictions(predictions[1, :, :], sd.shape, self.configs["is_old_roi"])

                min_area_threshold = 4000
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [contour for contour in contours if cv2.contourArea(contour) > min_area_threshold]
                contours_mask = np.zeros_like(sd)
                cv2.drawContours(contours_mask, contours, -1, 1, thickness=cv2.FILLED)
                sd[contours_mask == 0] = 0

                roi_area = 0
                for cnt in contours:
                    roi_area += cv2.contourArea(cnt)

                res = get_resolution(os.path.join(images_dir, filename + ".tif"), sd.shape[0])
                _, distances = self.skeleton_length(sd, res)
                total_sd_len = np.sum(distances)

                total_roi_area = roi_area * res ** 2

                sd_len = total_sd_len / total_roi_area
                grid_points, grid_index = self.calculate_grid(sd, res)

                if self.configs['does_include_sd']:
                    sd_grid_file.write(
                         "%s\t%.3f\t%i\t%.3f\t%.3f\t%.3f\n" % (
                            filename, sd_len, grid_points, grid_index, total_sd_len, total_roi_area))
        finally:
            if self.configs['does_include_sd']:
                sd_grid_file.close()

    @staticmethod
    def foot_process_parameters(region, res):
        region = region.astype(np.uint8)
        contours, hier = cv2.findContours(region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        area = cv2.contourArea(cnt) * res ** 2
        per = cv2.arcLength(cnt, True) * res
        if per == 0:
            return -1, -1, -1
        circ = (4 * math.pi * area) / (per ** 2)
        # if circ > 1:
        #     print(circ)
        return per, area, circ

    @staticmethod
    def get_number_of_neighbors(image, x, y):
        return np.sum(image[max(0, x - 1):min(image.shape[0], x + 2),
                      max(0, y - 1):min(image.shape[1], y + 2)]) - image[x, y]

    def tag_image(self, input_image):
        output_image = np.zeros_like(input_image)

        for x in range(input_image.shape[0]):
            for y in range(0, input_image.shape[1]):
                if input_image[x, y] > 0:
                    num_neighbors = self.get_number_of_neighbors(input_image, x, y)
                    if num_neighbors < 2:
                        output_image[x, y] = self.END_POINT
                    elif num_neighbors > 2:
                        output_image[x, y] = self.JUNCTION_POINT
                    else:
                        output_image[x, y] = self.SLAB_POINT
        return output_image

    def mark_trees(self, tagged_image, res):
        colored_image = np.zeros_like(tagged_image, dtype=int)
        visited_image = np.zeros_like(tagged_image)
        color = 0
        distances = []

        end_point_xs, end_point_ys = np.where(tagged_image == self.END_POINT)
        # Visit trees starting at end points
        for i in range(end_point_xs.size):
            x, y = end_point_xs[i], end_point_ys[i]
            if visited_image[x, y] == 0:
                colored_image, visited_image, dist = self.visit_tree(x, y,
                                                                     tagged_image,
                                                                     colored_image,
                                                                     visited_image,
                                                                     color,
                                                                     res)
                distances.append(dist)
                color += 1

        jun_point_xs, jun_point_ys = np.where(tagged_image == self.JUNCTION_POINT)
        for i in range(jun_point_xs.size):
            x, y = jun_point_xs[i], jun_point_ys[i]
            if visited_image[x, y] == 0:
                colored_image, visited_image, dist = self.visit_tree(x, y,
                                                                     tagged_image,
                                                                     colored_image,
                                                                     visited_image,
                                                                     color,
                                                                     res)
                distances.append(dist)
                color += 1

        # Check for unvisited slab voxels in case there are circular trees without junctions
        slab_point_xs, slab_point_ys = np.where(tagged_image == self.SLAB_POINT)
        for i in range(slab_point_xs.size):
            x, y = slab_point_xs[i], slab_point_ys[i]
            if visited_image[x, y] == 0:
                # Mark that voxel as the start point of the circularskeleton
                colored_image, visited_image, dist = self.visit_tree(x, y,
                                                                     tagged_image,
                                                                     colored_image,
                                                                     visited_image,
                                                                     color,
                                                                     res)
                distances.append(dist)
                if np.any(colored_image == color):
                    color += 1

        return colored_image, distances

    @staticmethod
    def find_unvisited(x, y, tagged_image, visited_image):
        for i in range(max(0, x - 1), min(visited_image.shape[0], x + 2)):
            for j in range(max(0, y - 1), min(visited_image.shape[1], y + 2)):
                if ((i != x) or (j != y)) and (visited_image[i, j] == 0) and (tagged_image[i, j] > 0):
                    return i, j
        return -1, -1

    @staticmethod
    def distance(x1, y1, x2, y2, res):
        dx = (x1 - x2) * res
        dy = (y1 - y2) * res
        return (dx ** 2 + dy ** 2) ** 0.5

    def visit_tree(self, x, y, tagged_image, colored_image, visited_image, color, res):
        colored_image[x, y] = color
        dist = 0
        to_revisit = []
        if tagged_image[x, y] == self.JUNCTION_POINT:
            to_revisit.append((x, y))

        next_x, next_y = self.find_unvisited(x, y, tagged_image, visited_image)
        prev_x, prev_y = x, y
        visited_image[prev_x, prev_y] = 1
        while (next_x >= 0) or (len(to_revisit) > 0):
            if next_x >= 0:
                visited_image[next_x, next_y] = 1
                colored_image[next_x, next_y] = color
                dist += self.distance(prev_x, prev_y, next_x, next_y, res)
                if tagged_image[next_x, next_y] == self.JUNCTION_POINT:
                    to_revisit.append((next_x, next_y))
                prev_x, prev_y = next_x, next_y
                next_x, next_y = self.find_unvisited(next_x, next_y, tagged_image, visited_image)
            else:
                prev_x, prev_y = to_revisit[0]
                next_x, next_y = self.find_unvisited(prev_x, prev_y, tagged_image, visited_image)
                if next_x < 0:
                    to_revisit.remove((prev_x, prev_y))
        return colored_image, visited_image, dist

    @staticmethod
    def take_middle_points(pts):
        res = [pts[0]]
        ds = pts[1:] - pts[:-1]
        last_non0 = 0
        for i in range(1, ds.size):
            if ds[i] != 1:
                # print(pts[(last_non0+1):(i+1)])
                res.append(np.mean(pts[(last_non0 + 1):(i + 1)]))
                last_non0 = i
        res.append(np.mean(pts[(last_non0 + 1):(pts.size)]))
        return np.array(res)

    def calculate_grid(self, sd, res):
        grid_d = 0.75 / res
        grid_steps = np.round(np.arange(0, sd.shape[0], grid_d)).astype(int)
        grid_steps = grid_steps[grid_steps < sd.shape[0]]
        all_ds = np.zeros(0)
        all_pts = 0
        for step in grid_steps:
            pts = np.where(sd[step, :] == 1)[0]
            if pts.size > 1:
                pts = self.take_middle_points(pts)
                all_pts += pts.size
                ds = pts[1:] - pts[:-1]
                ds = ds * res
                all_ds = np.append(all_ds, ds)
            pts = np.where(sd[:, step] == 1)[0]
            if pts.size > 1:
                pts = self.take_middle_points(pts)
                all_pts += pts.size
                ds = pts[1:] - pts[:-1]
                ds = ds * res
                all_ds = np.append(all_ds, ds)
        return all_pts, np.mean(all_ds)

    def combine_FP_SD(self, param_dr):
        if self.configs['does_include_sd']:
            t = pd.read_table(os.path.join(param_dr, "SD_length_grid_index.csv"))
        else:
            t = pd.DataFrame()
        foot_process_area = np.zeros((t.shape[0]))
        foot_process_perim = np.zeros((t.shape[0]))
        foot_process_circ = np.zeros((t.shape[0]))

        if self.configs['does_include_sd']:
            for i in range(t.shape[0]):
                fl = t["file"][i]
                fp_t = np.loadtxt(os.path.join(param_dr, fl + "_fp_params.csv"), delimiter="\t", skiprows=1, ndmin=2)
                if fp_t.size > 0:
                    foot_process_area[i] = np.mean(fp_t[:, 1])
                    foot_process_perim[i] = np.mean(fp_t[:, 2])
                    foot_process_circ[i] = np.mean(fp_t[:, 3])
                else:
                    foot_process_area[i] = 0
                    foot_process_perim[i] = 0
                    foot_process_circ[i] = 0
            t["FP Area"] = foot_process_area
            t["FP Perim."] = foot_process_perim
            t["FP Circ."] = foot_process_circ
        else:
            suffix = "_fp_params.csv"
            files = []
            for file_path in os.listdir(param_dr):
                if file_path.endswith(suffix):
                    filename = os.path.basename(file_path)
                    fl = filename.replace(suffix, '')
                    files.append(fl)

            foot_process_area = np.zeros(len(files))
            foot_process_perim = np.zeros(len(files))
            foot_process_circ = np.zeros(len(files))

            for i in range(len(files)):
                fl = files[i]
                fp_t = np.loadtxt(os.path.join(param_dr,
                                               fl + "_fp_params.csv"),
                                  delimiter="\t", skiprows=1, ndmin=2)
                if fp_t.size > 0:
                    foot_process_area[i] = np.mean(fp_t[:, 1])
                    foot_process_perim[i] = np.mean(fp_t[:, 2])
                    foot_process_circ[i] = np.mean(fp_t[:, 3])
                else:
                    foot_process_area[i] = 0
                    foot_process_perim[i] = 0
                    foot_process_circ[i] = 0

            t["file"] = files
            t["FP Area"] = foot_process_area
            t["FP Perim."] = foot_process_perim
            t["FP Circ."] = foot_process_circ

        t.to_csv(os.path.join(param_dr, "all_params.csv"), sep="\t")
