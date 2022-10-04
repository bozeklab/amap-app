# AMAP Application

AMAP is a desktop application that performs [segmentation and morphometry quantification of fluorescent microscopy images of podocytes](https://www.biorxiv.org/content/10.1101/2021.06.14.448284v3) on the CPU. Therefore, users don't need GPUs to use the application.

![AMAP Results](res/images/header.png)

This application is developed based on [AMAP](https://github.com/bozeklab/amap) repositery.

### Requirements

AMAP is implemented in Python 3 and tested in Linux using Python 3.7 and 3.10. A full list of required packages is available in [requirements.txt](./requirements.txt), but to name major dependencies:

* AMAP uses [Pyside6](https://pypi.org/project/PySide6/) for its user interface which is a Python wrapper for the Qt framework.
* [Pytorch](https://pytorch.org/) is used for training and inference of deep learning models.
* [tifffile](https://pypi.org/project/tifffile/) is used to read data samples in tiff format
* [scikit-learn](https://scikit-learn.org/stable/), [OpenCV](https://pypi.org/project/opencv-python/), and [NumPy](https://numpy.org/) are used for various image processing and machine learning tasks.
* [BICO](https://ls2-www.cs.tu-dortmund.de/grav/en/bico), a fast and efficient kmeans algorithm implemented in C++ used for clustering.

### Installation

AMAP application targeted to support Linux, macOS, and Windows. But platform-dependent implementation of the multi-processing module in Python hinders achieving that goal. For the time being, it is recommended to use the provided virtual machine(VM) to use the application. All the dependencies are installed in the VM and the application is ready to use.

Some virtualization software is needed to use the VM. The import steps are shown below for VirtualBox. Therefore, please install Virtual Box before proceeding. If you are using other virtualization software, the ova image can easily be imported into other virtualization solutions as well.

* macOS
* Windows
* Linux

### File Transfer: Host & VM

### Using the application