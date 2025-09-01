# Python Imports
import json
import os
import time
import logging.config

# Library Imports
import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as fn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Local Imports
from src.nn.dataset import PredictionDataset
from src.nn.unet import UNet
from src.utils import mkdirs, plot_labels
from src.configs import LOG_START_PROC_SIGNATURE
from src.utils import get_ROI_from_predictions


# Labels of the model output
BACKGROUND = 0
FOOTPROCESS = 1
SDLINE = 2


class AMAPEngine:
    def __init__(self, _configs):
        # Collecting HW info
        logging.info("Scanning the hardware")
        self.cpu_count = psutil.cpu_count()
        self.memory_size = psutil.virtual_memory().total // 1024 ** 2  # To get the result in MiB

        logging.info(f"No of logical cores: {self.cpu_count}")
        logging.info(f"Memory: {self.memory_size} MiB")

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
        self.cpu_alloc_value = _configs['cpu_allocation']
        # This will multiply with batch_size and slider value starts from 0, so we add 1
        self.mem_alloc_value = _configs['mem_allocation'] + 1
        self.is_stacked = _configs['is_stacked']
        self.target_channel = _configs['target_channel']

        # This variable is used to stop the engine
        self.proceed = mp.Value('i', 1)

        # Set batch size according to the memory consumption setting
        self.batch_size = self.batch_size * self.mem_alloc_value
        logging.info("Setting batch size to %d.", self.batch_size)

        # Set number of threads for PyTorch engine
        # https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
        threads_num = max(1, int((self.cpu_alloc_value-1) / 4 * self.cpu_count))
        logging.info("Using %d logical cores.", threads_num)
        torch.set_num_threads(threads_num)

        # Engine configuration variables
        self.TARGET_RESOLUTION = 0.022724609375
        self.SAMPLE_SIZE = 384
        self.DATASET_STEPS = 128
        self.MIN_PIXELS = 25
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

        # Might be used later
        # self.no_of_gpus = min(_args.gpus, len(gpus))
        # self.use_gpu = self.no_of_gpus > 0

        logging.info("Creating the dataset from the images.")
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
        logging.info(LOG_START_PROC_SIGNATURE)
        logging.info(f"Inference started for {self.project_name}")

        self.start_time = time.time()

        self.inference_procedure()

        self.end_time = time.time()

        spent_time = self.end_time - self.start_time
        hours, remainder = divmod(spent_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"Inference finished in: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.")

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

        logging.info(LOG_START_PROC_SIGNATURE)

        torch.manual_seed(0)

        logging.debug("Creating the Unet model.")
        # We use a single grayscale image as input, so n_channels=1
        # For the output 0=background, 1=footprocess, 2=SD Line
        # We do not use the embeddings in this version of the algorithm
        # but we need to pass the correct value(16) for the model to be
        # compatible with the snapshot.
        unet_model = UNet(n_channels=1,
                          n_classes=3,
                          n_dim=self.embedding_dimensionality,
                          bilinear=True)

        logging.debug("Moving the model to CPU.")
        device = torch.device('cpu')
        unet_model.to(device)
        unet_model.eval()

        logging.debug("Loading the checkpoint.")
        # model_checkpoint_path = "res/model/cp_10940.pth"
        model_checkpoint_path = "res/model/cp_11128.pth"
        unet_model.load_state_dict(torch.load(model_checkpoint_path,
                                              map_location=torch.device('cpu')))

        logging.debug("Creating the data loader.")
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            # It's important to turn off shuffle
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

        with torch.no_grad():
            # The dataset holds the patches for all the images in the project
            # but it is not randomized, so we are going through the images, while
            # we got through the patches, and the order of patches is the same as images
            for batch_i, batch in enumerate(loader):
                if not self.shall_proceed():
                    break

                images = batch['image']
                offsets = batch['offs']

                images = images.to(device)
                # The inferene happens here
                semantic_predictions, _ = unet_model(images)
                semantic_predictions = fn.softmax(semantic_predictions, dim=1)

                # Here we go through the patches in the batch and decide if it contain the
                # last patch of an image or not. If yes, we merge the patches
                # and if no, we store the result for future merging.
                for index, prediction in enumerate(semantic_predictions):

                    # The code is multio threaded, we need a locking mechanism for share resources
                    with self.processed_tiles.get_lock():
                        self.processed_tiles.value += 1.0

                    # To detect when we should merge patches of an image, we compare self.image_id with the
                    # image_id of the patch, this work for all the images except for the last one
                    # So, we should additionaly check for last patch as well.
                    end_condition = (batch_i == len(loader)-1) and (index == len(semantic_predictions)-1)
                    offset = offsets[index]

                    if offset[0] != self.image_id or end_condition:
                        # In this case, the inference for an image is finished,
                        # and we apply the CCL algorithm and store the result as a numpy file
                        # for morphometry engine to use
                        filepath = self.dataset.image_files[self.image_id]

                        logging.info(f"Merging patches for the file: {filepath}")
                        self.merge_patches()

                        image_size = self.dataset.image_shape_by_id(self.image_id)[1:]
                        image_size = (2, *image_size)
                        # mask_img contains both sematic and instance segmentation results
                        mask_img = np.zeros(image_size)
                        mask_img[1] = self.semantic_mask

                        cc_mask = self.semantic_mask == SDLINE
                        self.semantic_mask[cc_mask] = BACKGROUND

                        logging.info("Applying CCL on the results.")
                        cc_number, cc_objects = cv2.connectedComponents(self.semantic_mask.astype(np.uint8))

                        self.remove_small_and_on_border(cc_number, cc_objects)

                        mask_img[0] = cc_objects

                        npy_out_dir, _ = mkdirs(self.output_npy_directory, filepath)
                        sub_out_dir, fn_short = mkdirs(self.output_segmentation_directory, filepath)

                        numpy_file_path = os.path.join(npy_out_dir, "%s_pred.npy" % fn_short[:-4])
                        logging.info(f"Saving the results as numpy file: {numpy_file_path}")
                        np.save(numpy_file_path, mask_img)

                        result_file_path = os.path.join(sub_out_dir, "%s_pred.png" % fn_short[:-4])

                        roi_mask, _ = get_ROI_from_predictions(mask_img[1, :, :],
                                                               mask_img[1, :, :].shape,
                                                               self.configs['is_old_roi'])

                        min_area_threshold = 4000
                        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        contours = [contour for contour in contours if cv2.contourArea(contour) > min_area_threshold]

                        logging.info(f"Ploting the segmentation results as: {result_file_path}")
                        plot_labels(self.dataset.read_file(filepath),
                                    cc_objects,
                                    mask_img[1],
                                    contours,
                                    cc_number,
                                    result_file_path)

                        self.patches.clear()
                        self.image_id = offset[0]

                    # Here we add the patch to list of procesed patches
                    self.patches.append((offsets[index], np.argmax(semantic_predictions[index], axis=0)))

                logging.debug(f"Inference of the batch no: {batch_i} finished.")

            if self.shall_proceed():
                self.configs['is_segmentation_finished'] = True
                config_file_path = os.path.join(self.base_directory, "conf.json")
                with open(config_file_path, 'w+') as file:
                    file.write(json.dumps(self.configs))

        logging.info("Finished, Exiting...")

    def remove_small_and_on_border(self, _cc_number, _image):
        logging.debug("Removing objects on borders.")
        on_border = np.unique(np.concatenate(
            [np.unique(_image[:, 0]),
             np.unique(_image[0, :]),
             np.unique(_image[:, -1]),
             np.unique(_image[-1, :])]))
        on_border = on_border[on_border != 0]
        for i in on_border:
            _image[_image == i] = 0

        logging.debug(f"Removing objects smaller than {self.MIN_PIXELS} pixels.")
        for i in range(1, _cc_number):
            is_i = _image == i
            if np.sum(is_i) < self.MIN_PIXELS:
                _image[_image == i] = 0

    # Converts self.proceed to bool
    def shall_proceed(self) -> bool:
        with self.proceed.get_lock():
            return bool(self.proceed.value)

    def cancel(self) -> None:
        with self.proceed.get_lock():
            self.proceed.value = 0
