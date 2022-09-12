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
import torch.multiprocessing as mp
import torch.nn.functional as fn
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

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

        # Cancellation variable
        self.proceed = mp.Array(ctypes.c_bool, 1)
        self.proceed.acquire()
        self.proceed[0] = True
        self.proceed.release()

        logging.info(f"Loading project configuration")
        # Configuration
        self.configs = _configs
        self.project_id = _configs['project_id']
        self.project_name = _configs['project_name']
        self.batch_size = _configs['batch_size']
        self.embedding_dimensionality = _configs['dimensionality']
        self.source_directory = _configs['source_dir']
        self.output_segmentation_directory = _configs['result_segmentation_dir']
        self.output_morphometry_directory = _configs['result_morphometry_dir']
        self.resource_alloc_value = _configs['resource_allocation']
        self.is_stacked = _configs['is_stacked']
        self.target_channel = _configs['target_channel']
        self.clustering_precision = _configs['clustering_precision']

        # Engine configuration variables
        self.TARGET_RESOLUTION = 0.022724609375
        self.SAMPLE_SIZE = 384
        self.DATASET_STEPS = 128
        self.SHARED_MEMORY_SIZE = 20
        self.MIN_PIXELS = 10
        self.BICO_DIR = 'res/bico/'
        self.TEMP_DIR = self.source_directory + '/temp/'
        self.LOG_DIR = self.source_directory + '/log/'
        self.CC_SCALE = 4

        self.no_of_tile_processes, self.no_of_cluster_processes = \
            self.resource_value_to_process_no(self.resource_alloc_value)

        # Process IDs
        self.exec_id = mp.Array(ctypes.c_int64, 1)
        self.collector_id = mp.Array(ctypes.c_int64, 1)
        self.inference_id = mp.Array(ctypes.c_int64, 1)
        self.clustering_ids = mp.Array(ctypes.c_int64, self.no_of_cluster_processes)
        self.tiling_ids = mp.Array(ctypes.c_int64, self.no_of_tile_processes)

        # Processed tiles are needed to calculate completion percentage
        self.no_of_processed_tiles = mp.Array(ctypes.c_int64, 1)
        self.no_of_processed_tiles.acquire()
        self.no_of_processed_tiles[0] = 0
        self.no_of_processed_tiles.release()

        if not os.path.exists(self.output_segmentation_directory):
            os.mkdir(self.output_segmentation_directory)

        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)

        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        os.mkdir(self.TEMP_DIR)

        # Might be used later
        # self.no_of_gpus = min(_args.gpus, len(gpus))
        # self.use_gpu = self.no_of_gpus > 0

        logging.info(f"Creating the dataset from images.")
        # Creating the dataset
        self.dataset = PredictionDataset(_source_directory=self.source_directory,
                                         _target_resolution=self.TARGET_RESOLUTION,
                                         _sample_dimension=self.SAMPLE_SIZE,
                                         _steps=self.DATASET_STEPS)

        # Variables to track time
        self.start_time = None
        self.end_time = None

        logging.info(f"Preparing shared memory arrays")

        # Shared Memory for Multi-Processing #

        # These variables are tightly related
        # predictions_queue contains integers that are
        # shared memory array's indexes, if we put(index)
        # into the queue, it means that memory of that index is
        # free to be used, and get() method of the queue, will block
        # the caller till a memory slot becomes available
        self.shared_memory_queue = mp.Queue()

        dim = self.embedding_dimensionality
        h = self.SAMPLE_SIZE
        w = self.SAMPLE_SIZE

        self.finished_events = []

        # we use torch.multiprocessing.Array for our shared memory storage
        # They have a locking mechanism that we use for accessing the array
        self.semantic_predictions_sh_memory = [mp.Array(ctypes.c_float,
                                                        h * w)
                                               for _ in range(self.SHARED_MEMORY_SIZE)]

        # Also, we want to use the allocated memory as a normal numpy array
        # thus we create a numpy array using frombuffer function and reinterpret
        # the memory as a numpy array
        # So for each image, at the end, we would have an mp.Array and an np.Array
        # that refer to the same memory.
        self.semantic_predictions_sh_memory = [(shared_array,
                                                np.frombuffer(shared_array.get_obj(),
                                                              dtype=np.float32,
                                                              count=h * w).
                                                reshape([h, w]))
                                               for shared_array in self.semantic_predictions_sh_memory]

        # Same as above
        self.instance_predictions_sh_memory = [mp.Array(ctypes.c_float,
                                                        dim * h * w)
                                               for _ in range(self.SHARED_MEMORY_SIZE)]
        self.instance_predictions_sh_memory = [(shared_array,
                                                np.frombuffer(shared_array.get_obj(),
                                                              dtype=np.float32,
                                                              count=dim * h * w).
                                                reshape([dim, h, w]))
                                               for shared_array in self.instance_predictions_sh_memory]

        # Each image is divided into overlapping tiles
        # These offsets determine tiles borders
        self.offsets_sh_memory = [mp.Array(ctypes.c_float, 4) for _ in range(self.SHARED_MEMORY_SIZE)]
        self.offsets_sh_memory = [(shared_array,
                                   np.frombuffer(shared_array.get_obj(),
                                                 dtype=np.int32,
                                                 count=4))
                                  for shared_array in self.offsets_sh_memory]

        # Flagging the shared memory slots as available
        for i in range(self.SHARED_MEMORY_SIZE):
            self.shared_memory_queue.put(i)

        # Shared Memory for Multi-Processing #

        # We need a means of communication between the worker process
        # Below queue act as both a blocking mechanism that synchronizes
        # processes, also they are used to transfer data between processes

        # res_q in Original AMAP
        # This queue is consumed by the collector process, tile processes will send signals
        # using it, the signal data type is integer, and it is actually and index < SHARED_MEMORY_SIZE
        # referring to a memory slot in the shared array
        self.collector_queue = mp.Queue()

        # These flags are used for interprocess communication using collector_queue

        # inference processes send the signal to collector process "collector_queue.put(inference_process_finished)"
        # Also, collector process send the signal to tile processes "tile_queue.put(inference_process_finished)"
        self.INFERENCE_PROCESS_FINISHED = -1

        # tile processes send the signal to collector process "collector_queue.put(tile_process_finished)"
        self.TILE_PROCESS_FINISHED = -2

        # collector process send the signal to cluster processes "cluster_queue.put(collector_process_finished)"
        self.COLLECTOR_PROCESS_FINISHED = (-1, 0, 0, "", "", "", "", 0, 0, 0)

        # tile_q in Original AMAP
        # This queue is consumed by the tile processes,the inference process uses it to signal
        # tile processes.The signal data type is integer, and it is actually and index < SHARED_MEMORY_SIZE
        # referring to a memory slot in the shared array
        self.tile_queue = mp.Queue()

        # cluster_res_qs in Original AMAP
        # This queue is consumed by tile processes, cluster process uses it to signal tile processes.
        # There is a queue for every tile process, the signal contains (n, n_obj, silhouette)
        # ToDo: rename (n, n_obj, silhouette) variables and update this comment
        self.tile_clusters_queue_array = [mp.Queue() for _ in range(self.no_of_tile_processes)]

        # cluster_q in Original AMAP
        # This queue is consumed by cluster processes, tile processes use it to signal cluster processes.
        # In addition to the memory address, it contains other information about files address that contain
        # clustering data. This is subject to change, as in this application bico is going to be called
        # through a wrapper instead of terminal.
        # ToDo: Update the comment after implementing the wrappers
        self.cluster_queue = mp.Queue()

    def exec(self):
        self.exec_id.acquire()
        self.exec_id[0] = os.getpid()
        self.exec_id.release()

        executor_logger = self.get_logger("executor")

        executor_logger.info(LOG_START_PROC_SIGNATURE)
        executor_logger.info(f"Execution process started for {self.project_name}")
        executor_logger.info(f"Process ID: {self.exec_id[0]}")

        self.start_time = time.time()

        event_finished = mp.Event()

        collector_process = mp.Process(target=self.collector_procedure,
                                       args=(event_finished,))
        self.finished_events.append(event_finished)
        collector_process.start()
        executor_logger.info(f"Collector process started.")

        executor_logger.info(f"No of clustering processes: {self.no_of_cluster_processes}.")
        for cluster_process_id in range(self.no_of_cluster_processes):
            event_finished = mp.Event()
            cluster_process = mp.Process(target=self.cluster_procedure,
                                         args=(event_finished, cluster_process_id))
            self.finished_events.append(event_finished)
            cluster_process.start()
            executor_logger.info(f"Clustering process no: {cluster_process_id} started.")

        executor_logger.info(f"No of tiling processes: {self.no_of_tile_processes}.")
        for tile_process_id in range(self.no_of_tile_processes):
            event_finished = mp.Event()
            tiling_process = mp.Process(target=self.tiling_procedure,
                                        args=(event_finished, tile_process_id))
            self.finished_events.append(event_finished)
            tiling_process.start()
            executor_logger.info(f"Tiling process no: {tile_process_id} started.")

        inference_process = mp.Process(target=self.inference_procedure,
                                       args=(self.finished_events,))
        inference_process.start()
        executor_logger.info(f"Inference process started.")

        executor_logger.info(f"Waiting for inference process to finish.")
        if inference_process:
            inference_process.join()
        else:
            executor_logger.info(f"This should not happen, please contact the developer team.")

        self.end_time = time.time()

        spent_time = self.end_time - self.start_time
        hours, remainder = divmod(spent_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        if self.proceed[0]:
            self.configs['is_segmentation_finished'] = True
            config_file_path = os.path.join(self.source_directory, "conf.json")
            with open(config_file_path, 'w+') as file:
                file.write(json.dumps(self.configs))

        executor_logger.info(f"Clustering finished in: {int(hours)}:{int(minutes)}:{int(seconds)}.")

    def collector_procedure(self, _event_finished):
        self.collector_id.acquire()
        self.collector_id[0] = os.getpid()
        self.collector_id.release()

        collector_logger = self.get_logger("collector")

        collector_logger.info(LOG_START_PROC_SIGNATURE)
        collector_logger.info(f"Collector process started for {self.project_name}")
        collector_logger.info(f"Process ID: {self.collector_id[0]}")

        result_dictionary = {}
        no_of_finished_tile_processes = 0

        collector_logger.debug(f"Starting collector loop")

        iteration_counter = 0
        while True:
            iteration_counter += 1
            collector_logger.debug(f"Iteration: {iteration_counter}")

            # Check if the project has been canceled and set the finished events before halting in case it was.
            if not self.proceed[0]:
                collector_logger.debug(f"Project {self.project_name} has been canceled, setting finished events.")
                for _ in range(self.no_of_cluster_processes):
                    self.cluster_queue.put(self.COLLECTOR_PROCESS_FINISHED)
                _event_finished.set()
                collector_logger.debug(f"Halted, Exiting...")
                return

            collector_logger.debug(f"Calling collector_queue.get()")
            shared_memory_index = self.collector_queue.get()
            collector_logger.debug(f"Received shared memory index: {shared_memory_index}")

            if shared_memory_index == self.INFERENCE_PROCESS_FINISHED:
                collector_logger.debug(f"Inference process finished, sending the signal to tile queue")
                for _ in range(self.no_of_tile_processes):
                    self.tile_queue.put(self.INFERENCE_PROCESS_FINISHED)

            elif shared_memory_index == self.TILE_PROCESS_FINISHED:
                # tile process finished
                collector_logger.debug(f"A tiling process has finished.")

                no_of_finished_tile_processes += 1
                collector_logger.debug(f"{no_of_finished_tile_processes} tiling processes has finished till now.")
                if no_of_finished_tile_processes == self.no_of_tile_processes:
                    collector_logger.debug(f"All tiling processes finished, "
                                           f"sending {self.no_of_cluster_processes} signals to cluster queue.")
                    for _ in range(self.no_of_cluster_processes):
                        self.cluster_queue.put(self.COLLECTOR_PROCESS_FINISHED)

                    # finish all the clustering processes then exit
                    collector_logger.debug(f"Setting the finished event.")
                    _event_finished.set()

                    collector_logger.info(f"Finished, Exiting...")
                    return
            else:
                collector_logger.debug(f"Copying semantic and instance predictions to local variables.")
                semantic_prediction = np.copy(self.semantic_predictions_sh_memory[shared_memory_index][1])
                instance_prediction = np.copy(self.instance_predictions_sh_memory[shared_memory_index][1][0, :, :])
                offset = np.copy(self.offsets_sh_memory[shared_memory_index][1])
                image_id, x, y, d_img = tuple(offset)

                # Q: What is the use of n_pred?
                # n_pred = np.max(instance_prediction)
                # print("%i %i-%i (%i) -- %i clusters" % (image_number, x, y, shared_memory_index, n_pred), flush=True)

                # Put back this shared_memory_index into the shared memory queue
                # This means this shared memory can be used for next predictions
                collector_logger.debug(f"Freeing shared memory slot no: {shared_memory_index}")
                self.shared_memory_queue.put(shared_memory_index)

                # res dictionary combines predictions of individual tiles into full images
                if not (image_id in result_dictionary):
                    result_dictionary[image_id] = (np.zeros((2, d_img, d_img), dtype=int), 0)
                mask_img, count = result_dictionary[image_id]
                mask_img = self.merge_with_mask(mask_img, instance_prediction, semantic_prediction, x, y)
                count += 1

                self.no_of_processed_tiles.acquire()
                self.no_of_processed_tiles[0] += 1
                self.no_of_processed_tiles.release()

                # if we have all the tiles for this image, save results to files
                if count == self.dataset.n_per_img(image_id):
                    collector_logger.debug(f"Image with image_id: {image_id} finished, "
                                           f"Combining the tiles into a full image.")

                    filepath = self.dataset.image_files[image_id]
                    mask_inst = mask_img[0]
                    mask_sem = mask_img[1]
                    sub_out_dir, fn_short = mkdirs(self.output_segmentation_directory, filepath)

                    np.save(os.path.join(sub_out_dir, "%s_pred.npy" % fn_short[:-4]), mask_img)

                    result_file_name = os.path.join(sub_out_dir, "%s_pred.png" % fn_short[:-4])
                    collector_logger.debug(f"Saving the result to: {result_file_name}")

                    plot_labels(self.dataset.read_file(filepath),
                                mask_inst,
                                mask_sem,
                                np.max(mask_inst),
                                result_file_name)
                    del result_dictionary[image_id]
                else:
                    collector_logger.debug(
                        f"Adding mask image for image_id: {image_id} - {count}/{self.dataset.n_per_img(image_id)}")
                    result_dictionary[image_id] = (mask_img, count)

    @staticmethod
    def merge_with_mask(_image_mask, _instance_prediction, _semantic_prediction, _x, _y):
        max_label = np.max(_image_mask)
        prediction_labels = np.unique(_instance_prediction)
        prediction_labels = prediction_labels[prediction_labels > 0]
        mask_instance = _image_mask[0, _x:(_x + _instance_prediction.shape[0]), _y:(_y + _instance_prediction.shape[1])]
        mask_semantic = _image_mask[1, _x:(_x + _instance_prediction.shape[0]), _y:(_y + _instance_prediction.shape[1])]
        for label in prediction_labels:
            is_label = _instance_prediction == label
            mask_values = mask_instance[is_label]
            mask_ls, counts = np.unique(mask_values, return_counts=True)
            not_zero = mask_ls != 0
            enough_counts = (counts / np.sum(is_label)) > 0.1
            mask_ls = mask_ls[not_zero & enough_counts]
            counts = counts[not_zero & enough_counts]
            if mask_ls.size == 0:
                mask_instance[is_label] = max_label + 1
                max_label += 1
            else:
                mask_l = mask_ls[np.where(counts == np.max(counts))[0][0]]
                mask_instance[is_label] = mask_l

        # merge semantic segm - maximum of class
        mask_semantic = np.max(
            np.append(np.expand_dims(mask_semantic, 0),
                      np.expand_dims(_semantic_prediction, 0),
                      axis=0),
            axis=0)

        _image_mask[0, _x:(_x + _instance_prediction.shape[0]), _y:(_y + _instance_prediction.shape[1])] = mask_instance
        _image_mask[1, _x:(_x + _instance_prediction.shape[0]), _y:(_y + _instance_prediction.shape[1])] = mask_semantic
        return _image_mask

    def cluster_procedure(self, _event_finished, _cluster_process_id):
        self.clustering_ids.acquire()
        self.clustering_ids[_cluster_process_id] = os.getpid()
        self.clustering_ids.release()

        cluster_logger = self.get_logger(f"clustering_{_cluster_process_id}")

        cluster_logger.info(LOG_START_PROC_SIGNATURE)
        cluster_logger.info(f"Cluster process no: {_cluster_process_id} started for {self.project_name}")
        cluster_logger.info(f"Process ID: {self.clustering_ids[_cluster_process_id]}")

        iteration_counter = 0
        while True:
            iteration_counter += 1
            cluster_logger.debug(f"Iteration: {iteration_counter}")

            if not self.proceed[0]:
                cluster_logger.debug(f"Project {self.project_name} has been canceled, setting finished events.")
                _event_finished.set()
                cluster_logger.info("Halted, Exiting...")
                return
            (i, n_obj, n, cc_fl, emb_fl, cs_fl, out_fl, npoints, d, tile_nproc) = self.cluster_queue.get()
            if i == -1:
                cluster_logger.debug(f"All tiling process finished, setting finished event.")
                _event_finished.set()
                cluster_logger.info(f"Finished, Exiting...")
                return
            else:
                cluster_logger.debug(f"Calling cluster(part of bico) executable.")

                bico_call = os.path.join(self.BICO_DIR, "cluster")
                bico_call += " \"%s\" \"%s\" %i %i %i \"%s\" 5 &> /dev/null" % (
                    cs_fl, emb_fl, n_obj, d, npoints, out_fl)
                cluster_logger.debug(f"Command: {bico_call}")
                os.system(bico_call)
                emb = np.loadtxt(emb_fl, dtype=np.float32, delimiter=',')
                labs = np.loadtxt(out_fl, dtype=int)

                emb_no = len(emb)
                sample_size = 200
                if self.clustering_precision == 4:
                    cluster_logger.debug(f"Setting sample size of silhouette score to 20%")
                    sample_size = max(emb_no // 5, sample_size)
                elif self.clustering_precision == 3:
                    cluster_logger.debug(f"Setting sample size of silhouette score to 10%")
                    sample_size = max(emb_no // 10, sample_size)
                elif self.clustering_precision == 2:
                    cluster_logger.debug(f"Setting sample size of silhouette score to 5%")
                    sample_size = max(emb_no // 20, sample_size)
                elif self.clustering_precision == 1:
                    cluster_logger.debug(f"Setting sample size of silhouette score to 3.3%")
                    sample_size = max(emb_no // 30, sample_size)
                elif self.clustering_precision == 0:
                    cluster_logger.debug(f"Setting sample size of silhouette score to 2%")
                    sample_size = max(emb_no // 50, sample_size)

                cluster_score = silhouette_score(emb, labs, sample_size=sample_size)
                cluster_logger.debug(f"Cluster silhouette score: {cluster_score}")

                cluster_logger.debug(f"Saving the result into the tile_clusters_queue_array.")
                self.tile_clusters_queue_array[tile_nproc].put((n, n_obj, cluster_score))

    def inference_procedure(self, _finished_events):
        self.inference_id.acquire()
        self.inference_id[0] = os.getpid()
        self.inference_id.release()

        inference_logger = self.get_logger("inference")

        inference_logger.info(LOG_START_PROC_SIGNATURE)
        inference_logger.info(f"Inference process started for {self.project_name}")
        inference_logger.info(f"Process ID: {self.inference_id[0]}")

        torch.manual_seed(0)

        inference_logger.debug(f"The UNet has been create.")
        unet_model = UNet(n_channels=1,
                          n_classes=3,
                          n_dim=self.embedding_dimensionality,
                          bilinear=True)

        inference_logger.debug(f"Moving the model to CPU.")
        device = torch.device('cpu')
        unet_model.to(device)
        unet_model.eval()

        inference_logger.debug(f"Loading the checkpoint.")
        model_checkpoint_path = "res/model/cp_10940.pth"
        unet_model.load_state_dict(torch.load(model_checkpoint_path,
                                              map_location=torch.device('cpu')))

        inference_logger.debug(f"Creating the data loader.")
        sampler = torch.utils.data.RandomSampler(self.dataset)
        loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            sampler=sampler)

        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                if not self.proceed[0]:
                    inference_logger.debug(f"Project {self.project_name} has been canceled, setting finished events.")
                    self.collector_queue.put(self.INFERENCE_PROCESS_FINISHED)
                    inference_logger.info("Halted, Exiting...")
                    return
                images = batch['image']
                offsets = batch['offs']

                images = images.to(device)
                semantic_predictions, instance_predictions = unet_model(images)
                semantic_predictions = fn.softmax(semantic_predictions, dim=1)

                inference_logger.debug(f"Dispatching the batch no: {batch_i}.")
                self.dispatch(semantic_predictions.cpu().data.numpy(),
                              instance_predictions.cpu().data.numpy(),
                              offsets.cpu().data.numpy())

        inference_logger.info(f"Inference finished, sending the signal to collector_queue.")
        self.collector_queue.put(self.INFERENCE_PROCESS_FINISHED)

        inference_logger.info(f"Waiting for other processes to finish.")
        for event in _finished_events:
            event.wait()

        inference_logger.info(f"Finished, Exiting...")

    def tiling_procedure(self, _event_finished, _tile_process_id):

        self.tiling_ids.acquire()
        self.tiling_ids[_tile_process_id] = os.getpid()
        self.tiling_ids.release()

        tiling_logger = self.get_logger(f"titling_{_tile_process_id}")

        tiling_logger.info(LOG_START_PROC_SIGNATURE)
        tiling_logger.info(f"Tiling process no: {_tile_process_id} started for {self.project_name}")
        tiling_logger.info(f"Process ID: {self.tiling_ids[_tile_process_id]}")

        iteration_counter = 0
        while True:
            iteration_counter += 1
            tiling_logger.debug(f"Iteration: {iteration_counter}")

            tiling_logger.debug(f"Getting a new task. Means asking for a new shared memory index")
            shared_memory_index: int = self.tile_queue.get()
            tiling_logger.debug(f"Shared memory index: {shared_memory_index}")

            if not self.proceed[0]:
                tiling_logger.debug(f"Project {self.project_name} has been canceled, setting finished events.")

                tiling_logger.debug(f"Sending finished signal to collector queue")
                self.collector_queue.put(self.TILE_PROCESS_FINISHED)

                _event_finished.set()
                tiling_logger.info("Halted, Exiting...")

                return

            if shared_memory_index == self.INFERENCE_PROCESS_FINISHED:
                tiling_logger.debug(f"Inference is finished, so no new tile to process")
                tiling_logger.debug(f"Signaling collector process before exiting.")
                self.collector_queue.put(self.TILE_PROCESS_FINISHED)
                tiling_logger.debug(f"Setting finished event.")
                _event_finished.set()
                tiling_logger.debug(f"Finished, Exiting...")
                return
            else:
                tiling_logger.debug(f"Copying semantic and instance prediction.")

                semantic_mask = np.copy(self.semantic_predictions_sh_memory[shared_memory_index][1])

                instance_prediction = np.copy(self.instance_predictions_sh_memory[shared_memory_index][1])

                is_footProcessing = semantic_mask == 1

                tiling_logger.debug(f"Finding the connected components.")
                connectedComponents_number, connectedComponents_image \
                    = cv2.connectedComponents(is_footProcessing.astype(np.uint8))

                ind_sb, sb = self.is_small_on_border(connectedComponents_number, connectedComponents_image)
                semantic_mask[sb] = 0

                numberOfForegroundPixels = np.sum(semantic_mask == 1)
                tiling_logger.debug(f"No of foreground pixels: {numberOfForegroundPixels}")
                if numberOfForegroundPixels < self.MIN_PIXELS:
                    tiling_logger.debug(f"No of foreground pixels is less than minimum({self.MIN_PIXELS}).")
                    np.copyto(self.instance_predictions_sh_memory[shared_memory_index][1][0, :, :],
                              np.zeros((self.SAMPLE_SIZE, self.SAMPLE_SIZE)))
                    tiling_logger.debug(f"Ignoring by putting zeros instead of instance prediction.")
                    self.collector_queue.put(shared_memory_index)
                else:
                    inds = np.arange(0, connectedComponents_number)[np.logical_not(ind_sb)]
                    connectedComponents_image[sb] = 0
                    for j, ind in enumerate(inds):
                        connectedComponents_image[connectedComponents_image == ind] = j
                    connectedComponents_number = np.size(inds) - 1

                    if connectedComponents_number == 1:
                        np.copyto(self.instance_predictions_sh_memory[shared_memory_index][1][0, :, :],
                                  connectedComponents_image)
                        self.collector_queue.put(shared_memory_index)
                    else:
                        is_footProcessing = semantic_mask == 1
                        ccs = connectedComponents_image[is_footProcessing]

                        tiling_logger.debug(f"Adding one hot ccs to the embeddings.")
                        one_hot_ccs = np.eye(connectedComponents_number)[ccs - 1] * self.CC_SCALE
                        embeddings = instance_prediction[:, is_footProcessing].transpose(1, 0)
                        embeddings = np.append(one_hot_ccs, embeddings, axis=1)

                        cc_fl = os.path.join(self.TEMP_DIR, "%i_cc.txt" % shared_memory_index)
                        np.savetxt(cc_fl, ccs, fmt="%i", delimiter=",")

                        n = embeddings.shape[0]
                        emb_fl = os.path.join(self.TEMP_DIR, "%i.txt" % shared_memory_index)
                        np.savetxt(emb_fl, embeddings, fmt="%.4f", delimiter=",")

                        cs_fl = os.path.join(self.TEMP_DIR, "%i_coreset.txt" % shared_memory_index)
                        n_mean = connectedComponents_number
                        n_min, n_max = max(2, n_mean - 2), n_mean + 2

                        tiling_logger.debug(f"Calling BICO_Quickstart.")
                        bico_call = os.path.join(self.BICO_DIR, "BICO_Quickstart")
                        bico_call += " \"%s\" %i %i %i %i \"%s\" 10 , &>/dev/null" % (
                            emb_fl, n, n_mean,
                            self.embedding_dimensionality + connectedComponents_number, n // 2, cs_fl)
                        tiling_logger.debug(f"Command: {bico_call}")
                        os.system(bico_call)
                        while len(open(cs_fl).readlines()) <= n_max:
                            os.system(bico_call)

                        for n_obj in range(n_min, n_max + 1):
                            out_fl = os.path.join(self.TEMP_DIR, "%i_%i.txt" % (shared_memory_index, n_obj))
                            self.cluster_queue.put(
                                (shared_memory_index, n_obj, n_max - n_min + 1, cc_fl, emb_fl, cs_fl,
                                 out_fl, n,
                                 self.embedding_dimensionality + connectedComponents_number, _tile_process_id))

                        ranks = np.zeros((n_max - n_min + 1, 2))
                        for n_obj in range(n_min, n_max + 1):
                            if not self.proceed[0]:
                                tiling_logger.debug(
                                    f"Project {self.project_name} has been canceled, setting finished events.")

                                tiling_logger.debug(f"Sending finished signal to collector queue")
                                self.collector_queue.put(self.TILE_PROCESS_FINISHED)

                                _event_finished.set()
                                tiling_logger.info("Halted, Exiting...")
                                return

                            tiling_logger.debug(f"Getting clusters and scores "
                                                f"from tile_clusters_queue_array[_tile_process_id].get().")
                            (n, n_obj, silhouette) = self.tile_clusters_queue_array[_tile_process_id].get()
                            ranks[n_obj - n_min, :] = [n_obj, silhouette]

                        tiling_logger.debug(f"Choosing the best cluster.")
                        pred_inst, n_pred = self.pick_cluster(ranks, shared_memory_index, is_footProcessing)

                        # save to output shared array
                        tiling_logger.debug(f"Saving back instance mask to the shared memory.")
                        self.instance_predictions_sh_memory[shared_memory_index][0].acquire()
                        np.copyto(self.instance_predictions_sh_memory[shared_memory_index][1][0, :, :],
                                  pred_inst)
                        self.instance_predictions_sh_memory[shared_memory_index][0].release()

                        tiling_logger.debug(f"Send collector queue the signal "
                                            f"to collect the image no: {shared_memory_index}")
                        self.collector_queue.put(shared_memory_index)

                        tiling_logger.debug(f"Cleaning up the files.")
                        os.remove(cs_fl)
                        os.remove(cc_fl)
                        os.remove(emb_fl)
                        for n_obj in range(n_min, n_max + 1):
                            os.remove(os.path.join(self.TEMP_DIR, "%i_%i.txt" % (shared_memory_index, n_obj)))

    def dispatch(self, _semantic_predictions, _instance_predictions, _offsets):
        # Save predictions coming from the GPU to a free shared memory slot
        # then put information in the tile_queue for the tile processes to use it
        for i in range(_semantic_predictions.shape[0]):
            semantic_prediction = _semantic_predictions[i]
            semantic_prediction = np.argmax(semantic_prediction, axis=0)
            instance_prediction = _instance_predictions[i]
            offset = _offsets[i]

            if not self.proceed[0]:
                return

            shared_memory_index: int = self.shared_memory_queue.get()

            # copy to shared memory
            self.semantic_predictions_sh_memory[shared_memory_index][0].acquire()
            np.copyto(
                self.semantic_predictions_sh_memory[shared_memory_index][1][:, :],
                semantic_prediction)
            self.semantic_predictions_sh_memory[shared_memory_index][0].release()

            self.instance_predictions_sh_memory[shared_memory_index][0].acquire()
            np.copyto(
                self.instance_predictions_sh_memory[shared_memory_index][1][:, :, :],
                instance_prediction)
            self.instance_predictions_sh_memory[shared_memory_index][0].release()

            self.offsets_sh_memory[shared_memory_index][0].acquire()
            np.copyto(self.offsets_sh_memory[shared_memory_index][1][:], offset)
            self.offsets_sh_memory[shared_memory_index][0].release()

            self.tile_queue.put(shared_memory_index)

    def pick_cluster(self, ranks, i, is_fp):
        # pick the best cluster based on the collected statistics
        n_objs = ranks[:, 0]
        r = ranks[:, 1]
        i_max = np.where(r == np.max(r))[0][0]
        n_obj = int(n_objs[i_max])

        instance_mask = np.zeros_like(is_fp, dtype=int)

        out_fl = os.path.join(self.TEMP_DIR, "%i_%i.txt" % (i, n_obj))
        labs = np.loadtxt(out_fl, dtype=int)
        instance_mask[is_fp] += labs

        return instance_mask, n_obj

    def is_small_on_border(self, _connected_components_number, _image):
        on_border = np.unique(np.concatenate(
            [np.unique(_image[:, 0]),
             np.unique(_image[0, :]),
             np.unique(_image[:, -1]),
             np.unique(_image[-1, :])]))

        on_border = on_border[on_border != 0]
        ind = np.zeros(_connected_components_number, dtype=bool)
        res = np.zeros(_image.shape, dtype=bool)
        for i in on_border:
            res[_image == i] = True
            ind[i] = True

        for i in range(1, _connected_components_number):
            is_i = _image == i
            if np.sum(is_i) < self.MIN_PIXELS:
                res[_image == i] = True
                ind[i] = True

        return ind, res

    def cancel_processing(self):
        self.proceed.acquire()
        self.proceed[0] = False
        self.proceed.release()

        for _ in range(self.no_of_tile_processes):
            self.tile_queue.put(self.INFERENCE_PROCESS_FINISHED)

        for i in range(self.SHARED_MEMORY_SIZE):
            self.shared_memory_queue.put(i)

    # Decides how many tile and clustering processes to create
    # Based on user defined setting "resource allocation value"
    @staticmethod
    def resource_value_to_process_no(_resource_value):
        no_of_cpu = mp.cpu_count()

        def value_zero():
            return 1, 1

        def value_one():
            return 2, 2

        def value_two():
            return max(2, int(no_of_cpu / 2)), max(2, int(no_of_cpu / 2))

        def value_three():
            return max(2, no_of_cpu), max(2, no_of_cpu)

        option = {
            0: value_zero(),
            1: value_one(),
            2: value_two(),
            3: value_three()
        }
        return option[_resource_value]

    def get_logger(self, _process_name):
        logger = logging.getLogger(f"{self.project_id}-{_process_name}")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler = logging.FileHandler(f"{self.LOG_DIR}/{_process_name}.log")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(LOG_LEVEL)
        return logger
