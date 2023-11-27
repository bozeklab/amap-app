# AMAP Application

AMAP-APP is a desktop application that leverages deep learning to performs [segmentation and morphometry quantification of fluorescent microscopy images of podocytes](https://www.kidney-international.org/article/S0085-2538(23)00180-1/fulltext). The application is tailored for CPU utilization; GPUs are not requisite for its operation.

![AMAP Results](res/images/header.png)

This application represents a reimplementation of the [original reseasrch](https://github.com/bozeklab/amap) with modifications to the instance segmentation algorithm aimed at enhancing CPU efficiency. A notable departure from the original methodology is observed in the instance segmentation, which no longer depends on pixel embedding clustering. Instead, it capitalizes on PyTorch operations and a Connected Component Labeling algorithm to achieve comparable results.

AMAP-APP is a cross-platform application implemented in Python 3.9, primarily tested on Linux and with less extensive testing on Windows and Mac. While some visual inconsistencies may arise between different platforms, these variations do not compromise the functionality of the application.

## Requirements

#### Software
A full list of required packages is available in [requirements.txt](./requirements.txt), but to name major dependencies:

* AMAP uses [Pyside6](https://pypi.org/project/PySide6/) for its user interface which is a Python wrapper for the Qt framework.
* [Pytorch](https://pytorch.org/) is used for training and inference of the deep learning models.
* [tifffile](https://pypi.org/project/tifffile/) is used to read data samples in tiff format
* [scikit-learn](https://scikit-learn.org/stable/), [OpenCV](https://pypi.org/project/opencv-python/), and [NumPy](https://numpy.org/) are used for various image processing and machine learning tasks.

#### Hardware

Minimum: 4GB of RAM and 2 CPU cores.

Recommended: 16GB of RAM and 8 CPU cores.

## Installation

### Linux

1. Make sure you have below requirements installed
    * git
    * Python 3.9

2. Clone the repositary
```bash
git clone https://github.com/platonic-realm/amap-app.git
```

3. Preparing the Python environment
    * go into the **amap-app** directory

```bash
cd amap-app
```

    * Create a virtual environment

```bash
python -m venv venv
```

    * Activating the virtual environment

```bash
source ./venv/bin/activate
```

    * Instaling the requirements

```bash
pip instanll -r requirements.txt
```

## Update

To update AMAP to the latest version:

* Open a terminal in the amap-app directory

* Run the command:

```bash
git pull
```

## Using AMAP

AMAP application processes images in batches. A project is a batch of images combined with certain configurations. All images in a project must have the same order of dimensionality. At the current state AMAP only supports tiff files, therefore to create a project:

* Click on the "Add Project" button
* Select the directory containing the tiff files
* Configure the project
	* Resource Allocation: Determines how much of your computer resources will be allocated to the processing. The project finishes faster, but you might not be able to use your computer for other tasks if you choose high values.
	* Clustering Precision: AMAP uses scikit learn implementation of silhouette score for choosing the best number of clusters, which is very CPU intensitive. Reducing the sample size while calculating the silhouette score, accelerates the process by an order of magnitude without having serious effects on the results. Choosing high values will result in long processing times.
	* Target Channel: AMAP tries to automatically detect the target channel in the input images. Change this value when the automatic detection is wrong.
	* Stacked Checkbox: Determines whether the input images are an array of stacked images or not. If they are, AMAP will use a max projection of them. Change this value in the case of wrong detection.
* Click on the "Start" button and wait for the processing to finishes.
* Open Segmentation and Morphometry directories using the related buttons.
