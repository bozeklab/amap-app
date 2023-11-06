# Python Imports
import ctypes
import json
import os
import shutil
import time
import logging.config

# Library Imports
import cv2
import numpy as np
import psutil
import GPUtil
import torch
import torch.nn.functional as fn
from torch.utils.data import DataLoader
import torch,multiprocessing as mp
from PIL import Image

# Local Imports
from src.configs import LOG_LEVEL, LOG_START_PROC_SIGNATURE
from src.nn.dataset import PredictionDataset
from src.nn.unet import UNet
from src.utils import mkdirs, plot_labels


class AMAPEngine:
    def __init__(self, _configs):
        # Collecting HW info
        logging.info("Scanning the hardware")
        self.cpu_count = psutil.cpu_count()
        self.memory_size = psutil.virtual_memory().total // 1024 ** 2  # To get the result in MiB

        logging.info(f"No of logical cores: {self.cpu_count}")
        logging.info(f"Memory: {self.memory_size} MiB")

        # Collect info about system's GPU
        # GPUtil queries Nvidia GPUs info using nvidia-smi command
        self.list_gpus = []
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gid = gpu.id
            name = gpu.name
            memory = gpu.memoryTotal  # The value is in MB
            self.list_gpus.append({
                'id': gid,
                'name': name,
                'memory': memory,
            })

        logging.info("Loading project configuration")
        # Configuration
        self.configs = _configs
        self.project_id = _configs['project_id']
        self.project_name = _configs['project_name']
        self.batch_size = _configs['batch_size']
        self.embedding_dimensionality = _configs['dimensionality']
        self.base_directory = _configs['base_dir']
        self.source_directory = _configs['source_dir']
        self.output_segmentation_directory = _configs['result_segmentation_dir']
        self.output_npy_directory = _configs['npy_dir']
        self.output_morphometry_directory = _configs['result_morphometry_dir']
        self.resource_alloc_value = _configs['resource_allocation']
        self.is_stacked = _configs['is_stacked']
        self.target_channel = _configs['target_channel']
        self.clustering_precision = _configs['clustering_precision']

        # Engine configuration variables
        self.TARGET_RESOLUTION = 0.022724609375
        self.SAMPLE_SIZE = 384
        self.DATASET_STEPS = 128
        self.MIN_PIXELS = 20
        self.CC_SCALE = 4
        self.TEMP_DIR = self.base_directory + '/temp/'
        self.LOG_DIR = self.base_directory + '/log/'

        self.image_id = 0
        self.patches = []
        self.semantic_mask = None
        self.instance_mask = None
        self.processed_tiles = mp.Value('d', 0.0)

        if not os.path.exists(self.output_segmentation_directory):
            os.mkdir(self.output_segmentation_directory)

        if not os.path.exists(self.output_npy_directory):
            os.mkdir(self.output_npy_directory)

        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)

        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        os.mkdir(self.TEMP_DIR)

        # Might be used later
        # self.no_of_gpus = min(_args.gpus, len(gpus))
        # self.use_gpu = self.no_of_gpus > 0

        logging.info("Creating the dataset from images.")
        # Creating the dataset
        self.dataset = PredictionDataset(_configs=self.configs,
                                         _source_directory=self.source_directory,
                                         _target_resolution=self.TARGET_RESOLUTION,
                                         _sample_dimension=self.SAMPLE_SIZE,
                                         _steps=self.DATASET_STEPS)

        # Variables to track time
        self.start_time = None
        self.end_time = None

        logging.info("Preparing shared memory arrays")

    def exec(self):
        executor_logger = self.get_logger("executor")

        executor_logger.info(LOG_START_PROC_SIGNATURE)
        executor_logger.info(f"Execution process started for {self.project_name}")

        self.start_time = time.time()

        executor_logger.info("Waiting for inference to finish.")

        self.inference_procedure()

        self.end_time = time.time()

        spent_time = self.end_time - self.start_time
        hours, remainder = divmod(spent_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        executor_logger.info(f"Clustering finished in: {int(hours)}:{int(minutes)}:{int(seconds)}.")

    def merge_patches(self):
        image_size = self.dataset.image_shape_by_id(self.image_id)[1:]

        self.semantic_mask = np.zeros(image_size, dtype=int)
        self.instance_mask = np.zeros(image_size, dtype=int)
        for offset, patch in self.patches:
            id,  x, y, _ = offset
            patch_mask = self.semantic_mask[x:(x + patch.shape[0]), y:(y + patch.shape[1])]
            # merge semantic segm - maximum of class
            patch_mask = np.max(
                np.append(np.expand_dims(patch_mask, 0),
                          np.expand_dims(patch, 0),
                          axis=0),
                axis=0)
            self.semantic_mask[x:(x + patch.shape[0]), y:(y + patch.shape[1])] = patch_mask

    def inference_procedure(self):

        inference_logger = self.get_logger("inference")

        inference_logger.info(LOG_START_PROC_SIGNATURE)
        inference_logger.info(f"Inference process started for {self.project_name}")

        torch.manual_seed(0)

        inference_logger.debug("Creating the Unet model.")
        unet_model = UNet(n_channels=1,
                          n_classes=3,
                          n_dim=self.embedding_dimensionality,
                          bilinear=True)

        inference_logger.debug("Moving the model to CPU.")
        device = torch.device('cpu')
        unet_model.to(device)
        unet_model.eval()

        inference_logger.debug("Loading the checkpoint.")
        model_checkpoint_path = "res/model/cp_10940.pth"
        unet_model.load_state_dict(torch.load(model_checkpoint_path,
                                              map_location=torch.device('cpu')))

        inference_logger.debug("Creating the data loader.")
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                images = batch['image']
                offsets = batch['offs']

                images = images.to(device)
                semantic_predictions, _ = unet_model(images)
                semantic_predictions = fn.softmax(semantic_predictions, dim=1)

                for index, prediction in enumerate(semantic_predictions):
                    with self.processed_tiles.get_lock():
                        self.processed_tiles.value += 1.0

                    end_condition = (batch_i == len(loader)-1) and (index == len(semantic_predictions)-1)
                    offset = offsets[index]
                    if offset[0] != self.image_id or end_condition:
                        self.merge_patches()

                        image_size = self.dataset.image_shape_by_id(self.image_id)[1:]
                        image_size = (2, *image_size)
                        mask_img = np.zeros(image_size)
                        mask_img[0] = self.semantic_mask

                        cc_mask = self.semantic_mask == 2
                        self.semantic_mask[cc_mask] = 0

                        cc_number, cc_objects = cv2.connectedComponents(self.semantic_mask.astype(np.uint8))
                        self.remove_small_and_on_border(cc_number, cc_objects)

                        mask_img[1] = cc_objects

                        filepath = self.dataset.image_files[self.image_id]

                        npy_out_dir, _ = mkdirs(self.output_npy_directory, filepath)
                        sub_out_dir, fn_short = mkdirs(self.output_segmentation_directory, filepath)

                        np.save(os.path.join(npy_out_dir, "%s_pred.npy" % fn_short[:-4]), mask_img)

                        result_file_name = os.path.join(sub_out_dir, "%s_pred.png" % fn_short[:-4])

                        plot_labels(self.dataset.read_file(filepath),
                                    mask_img[1],
                                    mask_img[0],
                                    cc_number,
                                    result_file_name)

                        self.patches.clear()
                        self.image_id = offset[0]

                    self.patches.append((offsets[index], np.argmax(semantic_predictions[index], axis=0)))

                inference_logger.debug(f"Dispatching the batch no: {batch_i}.")

        inference_logger.info("Finished, Exiting...")

    def remove_small_and_on_border(self, _cc_number, _image):
        on_border = np.unique(np.concatenate(
            [np.unique(_image[:, 0]),
             np.unique(_image[0, :]),
             np.unique(_image[:, -1]),
             np.unique(_image[-1, :])]))
        on_border = on_border[on_border != 0]
        for i in on_border:
            _image[_image == i] = 0

        for i in range(1, _cc_number):
            is_i = _image == i
            if np.sum(is_i) < self.MIN_PIXELS:
                _image[_image == i] = 0

    def get_logger(self, _process_name):
        logger = logging.getLogger(f"{self.project_id}-{_process_name}")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler = logging.FileHandler(f"{self.LOG_DIR}/{_process_name}.log")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(LOG_LEVEL)
        return logger
