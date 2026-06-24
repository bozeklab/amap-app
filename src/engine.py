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
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Local Imports
from src.nn.dataset import PredictionDataset
from src.nn.unet import UNet
from src.utils import mkdirs, plot_labels
from src.configs import LOG_START_PROC_SIGNATURE
from src.utils import get_ROI_from_predictions, get_resolution, cpu_threads_from_level


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
        self.model_checkpoint = _configs.get('model_checkpoint', 'original.pth')
        self.use_gpu = _configs.get('use_gpu', True)
        self.num_workers = _configs.get('num_workers', 4)

        # This variable is used to stop the engine
        self.proceed = mp.Value('i', 1)

        # Set batch size according to the memory consumption setting
        self.batch_size = self.batch_size * self.mem_alloc_value
        logging.info("Setting batch size to %d.", self.batch_size)

        # Set number of threads for PyTorch engine
        # https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
        threads_num = cpu_threads_from_level(self.cpu_alloc_value, self.cpu_count)
        logging.info("Using %d logical cores.", threads_num)
        torch.set_num_threads(threads_num)

        # Engine configuration variables
        self.TARGET_RESOLUTION = 0.022724609375
        self.SAMPLE_SIZE = 384
        self.DATASET_STEPS = 128
        self.MIN_PIXELS = _configs.get('min_fp_pixels', 25)
        self.CC_SCALE = 4
        self.TEMP_DIR = self.base_directory + '/temp/'
        self.LOG_DIR = self.base_directory + '/log/'

        self.image_id = 0
        self.patches = []
        self.semantic_mask = None
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

    def _finalise_current_image(self):
        if not self.patches:
            return

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

        self.remove_small_and_on_border(cc_number, cc_objects,
                                        os.path.join(self.source_directory,
                                                     filepath))
        # self.fill_out_holes(cc_number, cc_objects)

        mask_img[0] = cc_objects

        npy_out_dir, _ = mkdirs(self.output_npy_directory, filepath)
        sub_out_dir, fn_short = mkdirs(self.output_segmentation_directory, filepath)

        numpy_file_path = os.path.join(npy_out_dir, "%s_pred.npy" % fn_short[:-4])
        logging.info(f"Saving the results as numpy file: {numpy_file_path}")
        np.save(numpy_file_path, mask_img)

        result_file_path = os.path.join(sub_out_dir, "%s_pred.png" % fn_short[:-4])

        roi_mask, _ = get_ROI_from_predictions(mask_img[1, :, :],
                                               mask_img[1, :, :].shape,
                                               self.configs['is_old_roi'],
                                               dilation_iters=self.configs.get('roi_dilation_iters', 25),
                                               erosion_iters=self.configs.get('roi_erosion_iters', 8),
                                               min_area=self.configs.get('roi_min_area', 5000))

        min_area_threshold = self.configs.get('roi_contour_min_area', 4000)
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

    def merge_patches(self):
        image_size = self.dataset.image_shape_by_id(self.image_id)[1:]

        self.semantic_mask = np.zeros(image_size, dtype=int)
        for offset, patch in self.patches:
            id,  x, y, _ = offset
            region = self.semantic_mask[x:(x + patch.shape[0]), y:(y + patch.shape[1])]
            # Where patches overlap, keep the higher-priority class per pixel.
            np.maximum(region, patch, out=region)

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

        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if (self.use_gpu and cuda_available) else "cpu")
        logging.info(
            "GPU toggle: %s, CUDA available: %s, using %s.",
            self.use_gpu, cuda_available, device)
        unet_model.to(device)
        unet_model.eval()

        logging.info("Loading the checkpoint.")
        model_checkpoint_path = os.path.join("res/model", self.model_checkpoint)
        unet_model.load_state_dict(torch.load(model_checkpoint_path,
                                               map_location=torch.device('cpu')))

        logging.debug("Creating the data loader.")
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            # It's important to turn off shuffle
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True)

        with torch.inference_mode():
            # The dataset holds the patches for all the images in the project
            # but it is not randomized, so we are going through the images, while
            # we got through the patches, and the order of patches is the same as images
            for batch_i, batch in enumerate(loader):
                if not self.shall_proceed():
                    break

                images = batch['image']
                offsets = batch['offs']

                filepath = self.dataset.image_files[self.image_id]
                logging.info(f"Segmentaion batch id: {batch_i} for the file: {filepath}")
                images = images.to(device)
                # The inferene happens here
                semantic_predictions, _ = unet_model(images)
                # argmax is invariant under softmax, so we skip the softmax and
                # pick the winning class per pixel on-device. Moving the small
                # [B, H, W] integer labels to the host once per batch is far
                # cheaper than copying the full [B, C, H, W] tensor per patch.
                batch_labels = torch.argmax(semantic_predictions, dim=1).cpu().numpy()

                # Here we go through the patches in the batch and decide if it contain the
                # last patch of an image or not. If yes, we merge the patches
                # and if no, we store the result for future merging.
                for index in range(batch_labels.shape[0]):

                    # Inference runs in a worker thread, so guard the shared counter.
                    with self.processed_tiles.get_lock():
                        self.processed_tiles.value += 1.0

                    offset = offsets[index]

                    # Whenever the batch crosses into a new image, finalise the
                    # previous one before accumulating patches for the new one.
                    if offset[0] != self.image_id:
                        self._finalise_current_image()
                        self.image_id = offset[0]

                    self.patches.append((offsets[index], batch_labels[index]))

                logging.debug(f"Inference of the batch no: {batch_i} finished.")

            # The loader is exhausted — finalise the patches accumulated for
            # the last image (they would otherwise sit in self.patches forever).
            if self.shall_proceed() and self.patches:
                self._finalise_current_image()

            if self.shall_proceed():
                self.configs['is_segmentation_finished'] = True
                config_file_path = os.path.join(self.base_directory, "conf.json")
                with open(config_file_path, 'w+') as file:
                    file.write(json.dumps(self.configs))

        logging.info("Finished, Exiting...")

    def fill_concave_regions_convex_hull(self, _cc_number, _image):
        """
        Fill concave regions by computing convex hull of each component.
        This will fill indentations and create smooth, convex shapes.
        """
        logging.debug("Filling concave regions using convex hull method.")

        for component_id in range(1, _cc_number):
            component_mask = (_image == component_id).astype(np.uint8)

            if np.sum(component_mask) == 0:
                continue

            # Find contours of the component
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Work with the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute convex hull
            hull = cv2.convexHull(largest_contour)

            # Create a mask from the convex hull
            hull_mask = np.zeros_like(component_mask)
            cv2.fillPoly(hull_mask, [hull], 1)

            # Fill the concave regions (pixels inside hull but outside original shape)
            newly_filled = (hull_mask == 1) & (component_mask == 0)
            _image[newly_filled] = component_id

    def fill_out_holes(self, _cc_number, _image):
        """
        Fill holes in connected components using OpenCV's morphological operations.

        Args:
            _cc_number: Number of connected components
            _image: The image containing connected components (modified in-place)
        """
        logging.debug("Filling holes in connected components.")

        # Process each connected component individually
        for component_id in range(1, _cc_number):
            # Create a binary mask for the current component
            component_mask = (_image == component_id).astype(np.uint8)

            # Skip if component doesn't exist (might have been removed by previous operations)
            if np.sum(component_mask) == 0:
                continue

            # Find contours of the component
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Fill holes by drawing filled contours
            if contours:
                # Create a temporary mask to fill holes
                filled_mask = np.zeros_like(component_mask)
                cv2.drawContours(filled_mask, contours, -1, 1, thickness=-1)  # Fill all contours

                # Find the newly filled pixels (holes that were filled)
                newly_filled = (filled_mask == 1) & (component_mask == 0)

                # Set the newly filled pixels to the component ID in the original image
                _image[newly_filled] = component_id

    def remove_small_and_on_border(self, _cc_number, _image, _file_path):
        logging.debug("Removing objects on borders.")
        on_border = np.unique(np.concatenate(
            [np.unique(_image[:, 0]),
             np.unique(_image[0, :]),
             np.unique(_image[:, -1]),
             np.unique(_image[-1, :])]))
        on_border = on_border[on_border != 0]
        if on_border.size:
            _image[np.isin(_image, on_border)] = 0

        # res = get_resolution(_file_path, _image.shape[1])
        # min_pix = 0.1 / (res ** 2)
        # logging.info(f"Removing objects smaller than {min_pix} pixels.")
        # Count every label in one pass and zero out the components smaller than
        # MIN_PIXELS via a label->keep lookup, instead of an O(components x size)
        # scan per label. Equivalent to the old per-label loop.
        sizes = np.bincount(_image.ravel())
        remove = sizes < self.MIN_PIXELS
        remove[0] = False  # never remove the background label
        _image[remove[_image]] = 0

    # Converts self.proceed to bool
    def shall_proceed(self) -> bool:
        with self.proceed.get_lock():
            return bool(self.proceed.value)

    def cancel(self) -> None:
        with self.proceed.get_lock():
            self.proceed.value = 0
