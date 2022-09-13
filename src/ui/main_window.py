# Python Imports
import json
import logging
import os
import pathlib
import shutil
import uuid

# Library Imports
import torch.multiprocessing as mp
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QIcon
from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QFileDialog, QMessageBox, QSpinBox, QListWidget, \
    QSlider, QCheckBox, QApplication

# Local Imports
from src.configs import PROJECT_DIR, HEADER_IMAGE, APP_ICON
from src.engine import AMAPEngine
from src.morph import AMAPMorphometry
from src.ui.ui_mainwindow import Ui_MainWindow
from src.utils import filter_tiff_files, analyze_tiff_files, create_progress_dialog, create_message_box, \
    open_dir_in_browser


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # This variable is used to prevent unnecessary writes while
        # configuring the projects
        self.is_loading = False
        # This variable is used to prevent signals causing problem
        # when UI elements are disabled
        self.is_disabled = True
        # Prevents double trigger of the timer
        self.is_triggered = False
        self.timer_time_out = 1000

        # Will be instantiated when start button is clicked
        self.engine = None
        self.morphometry = None
        self.progress_dialog = None
        self.project_process = None
        self.UI_state = None

        # These are ui elements that will be set in the configure_ui method
        self.app_icon = None
        self.button_add_project = None
        self.button_remove_project = None
        # This is a spin box to determine which channel is the tiff files are targeted
        self.spin_channel = None
        # This is a checkbox that determine whether the tiff images are stacked or not
        self.check_stacked = None
        # This is a QListWidget that demonstrates projects list
        self.list_projects = None
        # This is a QSlider for resource allocation configuration
        self.slider_resource = None
        # This is a QSlider for clustering precision configuration
        self.slider_precision = None

        # We need to access these labels later
        self.label_resource = None
        self.label_precision = None
        self.label_channel = None
        self.label_results = None

        # Start, Stop & Results buttons
        self.button_start = None
        self.button_stop = None
        self.button_results_segmentation = None
        self.button_results_morphometry = None

        # Ui_MainWindow class is autogenerated using "pyside6-uic" command
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.configure_ui()

    def configure_ui(self):

        self.setWindowTitle("AMAP")

        label = self.findChild(QLabel, "label_header")
        # The "Z003" font should be added to QFontDatabase before calling this method
        label.setFont(QFont("Z003", 36))

        label = self.findChild(QLabel, "label_image")
        label.setPixmap(QPixmap(HEADER_IMAGE))

        label = self.findChild(QLabel, "label_projects")
        label.setFont(QFont("Times", 14))

        label = self.findChild(QLabel, "label_resource_allocation")
        label.setFont(QFont("Times", 14))

        label = self.findChild(QLabel, "label_clustering_precision")
        label.setFont(QFont("Times", 14))

        label = self.findChild(QLabel, "label_channel")
        label.setFont(QFont("Times", 14))

        label = self.findChild(QLabel, "label_results")
        label.setFont(QFont("Times", 14))

        # These are the labels that we need to access them later
        self.label_resource = self.findChild(QLabel, "label_resource_allocation")
        self.label_precision = self.findChild(QLabel, "label_clustering_precision")
        self.label_channel = self.findChild(QLabel, "label_channel")
        self.label_results = self.findChild(QLabel, "label_results")

        # Setting the window icon
        self.app_icon = QIcon(APP_ICON)
        self.setWindowIcon(self.app_icon)

        # Configuring Start, Stop & Results buttons
        self.button_start = self.findChild(QPushButton, "button_start")
        self.button_start.clicked.connect(self.start_project_click)

        self.button_stop = self.findChild(QPushButton, "button_stop")
        self.button_stop.clicked.connect(self.stop_project_click)

        self.button_results_segmentation = self.findChild(QPushButton, "button_result_segmentation")
        self.button_results_segmentation.clicked.connect(self.segmentation_result_click)

        self.button_results_morphometry = self.findChild(QPushButton, "button_result_morphometry")
        self.button_results_morphometry.clicked.connect(self.morphometry_result_click)

        # Handling Add Project button's click
        self.button_add_project = self.findChild(QPushButton, "button_add_project")
        self.button_add_project.clicked.connect(self.add_project_click)

        # Handling Remove Project button's click
        self.button_remove_project = self.findChild(QPushButton, "button_remove_project")
        self.button_remove_project.clicked.connect(self.remove_project)

        # Handling spin box changes
        self.spin_channel = self.findChild(QSpinBox, "spin_channel")
        self.spin_channel.valueChanged.connect(self.spin_channel_change)

        # Handling resource slider changes
        self.slider_resource = self.findChild(QSlider, "slider_resource_allocation")
        self.slider_resource.valueChanged.connect(self.slider_resource_change)

        # Handling precision slider changes
        self.slider_precision = self.findChild(QSlider, "slider_precision")
        self.slider_precision.valueChanged.connect(self.slider_precision_change)

        # Handling stacked checkbox changes
        self.check_stacked = self.findChild(QCheckBox, "check_stacked")
        self.check_stacked.stateChanged.connect(self.checkbox_stack_change)

        # Handling list of projects
        self.list_projects = self.findChild(QListWidget, "list_widget_projects")
        self.list_projects.setFont(QFont("Times", 18))
        self.list_projects.itemClicked.connect(self.activate_selected_project)

        self.load_projects()

    # Load project configurations and enables the UI buttons
    def activate_selected_project(self, _selected_item):
        self.is_loading = True

        project_name = _selected_item.text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)

        self.slider_resource.setValue(project_configs['resource_allocation'])
        self.slider_resource.setEnabled(True)
        self.label_resource.setEnabled(True)

        self.slider_precision.setValue(project_configs['clustering_precision'])
        self.slider_precision.setEnabled(True)
        self.label_precision.setEnabled(True)

        self.spin_channel.setValue(project_configs['target_channel'])
        self.spin_channel.setEnabled(True)

        self.check_stacked.setChecked(project_configs['is_stacked'])
        self.check_stacked.setEnabled(True)
        self.label_channel.setEnabled(True)

        self.label_results.setEnabled(True)
        self.button_results_segmentation.setEnabled(project_configs['is_segmentation_finished'])
        self.button_results_morphometry.setEnabled(project_configs['is_morphometry_finished'])

        self.button_start.setEnabled(not project_configs['is_morphometry_finished'])
        self.button_stop.setEnabled(False)

        self.button_remove_project.setEnabled(True)

        self.is_disabled = False
        self.is_loading = False

    # Changes the channel configuration for the selected project
    def spin_channel_change(self, _value):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)
        project_configs['target_channel'] = _value
        self.save_project_configuration(project_configs_path, project_configs)

    # Changes the stacked configuration for the selected project
    def checkbox_stack_change(self, _value):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)
        project_configs['is_stacked'] = True if _value == 2 else False
        self.save_project_configuration(project_configs_path, project_configs)

    # Changes the resource configuration for the selected project
    def slider_resource_change(self, _value):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)
        project_configs['resource_allocation'] = _value
        self.save_project_configuration(project_configs_path, project_configs)

    # Changes the precision configuration for the selected project
    def slider_precision_change(self, _value):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)
        project_configs['clustering_precision'] = _value
        self.save_project_configuration(project_configs_path, project_configs)

    def stop_project_click(self):
        self.button_stop.setEnabled(False)
        self.engine.cancel_processing()
        self.progress_dialog.setLabelText(f'Cancelling... it will take some time.')
        self.engine = None

    def start_project_click(self):
        self.UI_state = self.save_UI_state()
        self.list_projects.setEnabled(False)
        self.disable_UI()
        self.button_add_project.setEnabled(False)
        self.button_stop.setEnabled(True)

        self.progress_dialog = create_progress_dialog(f'Segmentation', 'Please wait', self)

        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)

        if not project_configs['is_segmentation_finished']:
            self.engine = AMAPEngine(project_configs)
            self.project_process = mp.Process(target=self.start_project_segmentation)
            self.project_process.start()

        QtCore.QTimer.singleShot(500, self.check_project_status)

        self.progress_dialog.show()

    def check_project_status(self):
        if self.is_triggered:
            return
        else:
            self.is_triggered = True
        proceed = True
        try:
            if self.project_process is not None and self.project_process.is_alive():
                if self.engine is not None:
                    percentage = self.engine.no_of_processed_tiles[0] / len(self.engine.dataset) * 100
                    # A bug in Qt prevent us to set the percentage to 100
                    if percentage > 99:
                        percentage = 99
                    self.progress_dialog.setValue(percentage)
                elif self.morphometry is not None:
                    percentage = (self.morphometry.no_of_processed_images[0]) / self.morphometry.no_of_images[0] * 100
                    if percentage > 99:
                        percentage = 99
                    self.progress_dialog.setValue(percentage)

            else:
                project_name = self.list_projects.currentItem().text()
                project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
                project_configs = self.load_project_configuration(project_configs_path)

                if project_configs['is_segmentation_finished'] and not project_configs['is_morphometry_finished']:
                    # Segmentation Finished
                    if self.morphometry is None:
                        self.button_stop.setEnabled(False)
                        self.morphometry = AMAPMorphometry(project_configs)
                        self.project_process = mp.Process(target=self.start_project_morphometry)
                        self.progress_dialog.setLabelText(f'Morphometry')
                        self.project_process.start()

                else:
                    # Cancelled or finished
                    proceed = False

                    self.progress_dialog.close()
                    self.restore_UI_state(self.UI_state)
                    self.is_disabled = False
                    self.button_stop.setEnabled(False)
                    self.button_start.setEnabled(False)
                    self.button_results_morphometry.setEnabled(True)
                    self.button_results_segmentation.setEnabled(True)

                    self.UI_state = None
                    self.project_process = None
                    self.progress_dialog = None
                    self.engine = None
                    self.morphometry = None

        finally:
            self.is_triggered = False
            if proceed:
                QtCore.QTimer.singleShot(self.timer_time_out, self.check_project_status)

    def start_project_morphometry(self):
        if self.morphometry:
            self.morphometry.exec()

    def start_project_segmentation(self):
        if self.engine:
            self.engine.exec()

    def segmentation_result_click(self):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)

        open_dir_in_browser(project_configs['result_segmentation_dir'])

    def morphometry_result_click(self):
        if self.is_disabled or self.is_loading:
            return
        project_name = self.list_projects.currentItem().text()
        project_configs_path = f'./{PROJECT_DIR}/{project_name}/conf.json'
        project_configs = self.load_project_configuration(project_configs_path)

        open_dir_in_browser(project_configs['result_morphometry_dir'])

    def add_project_click(self):
        selected_directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if selected_directory == '':
            return

        selected_path = pathlib.PurePath(selected_directory)
        app_path = pathlib.PurePath(os.getcwd())

        # Checking for the cases that user can select a wrong path
        # The case that the project directory itself is selected
        if selected_path.parent.name == app_path.name:
            msgBox = create_message_box("You should not select projects directory. That's naughty!\n"
                                        "Please select another directory.",
                                        self.app_icon)
            msgBox.exec()
            return

        # The case that one of the directories inside project is selected
        if selected_path.parent.parent.name == app_path.name:
            msgBox = create_message_box("This project is already imported.\n"
                                        "Please select another directory.",
                                        self.app_icon)
            msgBox.exec()
            return

        # The case that no tif/tiff file exists in the selected directory
        tiff_files = filter_tiff_files(selected_directory)
        if len(tiff_files) == 0:
            msgBox = create_message_box("The selected directory doesn't contain any tif/tiff images.\n"
                                        "Currently, AMAP only support tif/tiff format.\n"
                                        "Please select another directory.",
                                        self.app_icon)
            msgBox.exec()
            return

        # The case that project already exits
        if os.path.exists(f"{os.getcwd()}/{PROJECT_DIR}/{selected_path.name}"):
            msgBox = create_message_box("The project already exists.\n"
                                        "Please remove the old one or rename the new one.",
                                        self.app_icon)
            msgBox.exec()
            return

        # Copying the selected directory and images to the project
        destination_directory = f"{os.getcwd()}/{PROJECT_DIR}/{selected_path.name}/"

        progressDialog = create_progress_dialog(f'Creating project "{selected_path.name}"', 'Please wait', self)
        progressDialog.show()
        QApplication.processEvents()

        # Checking the dimensionality of the images.
        # It should be the same for all images that belong to a project
        is_valid, is_stacked, dimensions = analyze_tiff_files(selected_directory)

        if not is_valid:
            progressDialog.close()
            logging.error('The shape of the last image is different from the rest.\n' + dimensions)
            msgBox = create_message_box("The dimension of all images in a project should be the same. "
                                        "Please review the output of the application in terminal for more info.",
                                        self.app_icon)
            msgBox.exec()
            return

        os.mkdir(destination_directory)

        images_directory = os.path.join(destination_directory, 'images/')
        os.mkdir(images_directory)

        for tiff_image in tiff_files:
            # Prevents UI freeze
            QApplication.processEvents()

            full_name = os.path.join(selected_directory, tiff_image)
            if os.path.isfile(full_name):
                shutil.copy(full_name, images_directory)

        # Saving the configuration file into the project directory (Do not confuse with 'Projects' dir)
        project_configuration = {
            "project_id": f"{uuid.uuid4()}",
            "project_name": f"{selected_path.name}",
            "base_dir": f"./{PROJECT_DIR}/{selected_path.name}/",
            "source_dir": f"./{PROJECT_DIR}/{selected_path.name}/images/",
            "npy_dir": f"./{PROJECT_DIR}/{selected_path.name}/npy/",
            "result_segmentation_dir": f"./{PROJECT_DIR}/{selected_path.name}/segmentation/",
            "result_morphometry_dir": f"./{PROJECT_DIR}/{selected_path.name}/morphometry/",
            "resource_allocation": 3,
            "clustering_precision": 3,
            "target_channel": 0,
            "batch_size": 2,
            "dimensionality": 16,
            "is_stacked": is_stacked,
            "is_segmentation_finished": False,
            "is_morphometry_finished": False
        }
        self.save_project_configuration(f"{destination_directory}/conf.json", project_configuration)
        self.load_projects()

        progressDialog.close()

        logging.info(f'Project "{selected_path.name}" created.')

    def remove_project(self):
        project_name = self.list_projects.currentItem().text()
        questionBox = QMessageBox()
        questionBox.setWindowIcon(self.app_icon)
        questionBox.setFont(QFont("Times", 14))
        answer = questionBox.question(self,
                                      '',
                                      f'The project "{project_name}" will be removed."\nAre you sure?',
                                      questionBox.Yes | questionBox.No)
        if answer == questionBox.Yes:
            def delete_project_dir(_project_name):
                project_path = f"./{PROJECT_DIR}/{_project_name}/"
                if os.path.exists(project_path):
                    shutil.rmtree(project_path)

            delete_project_dir(project_name)
            self.load_projects()
            self.list_projects.clearSelection()
            self.disable_UI()

            logging.info(f'Project "{project_name}" removed.')

    def disable_UI(self):
        self.slider_resource.setEnabled(False)
        self.label_resource.setEnabled(False)
        self.slider_precision.setEnabled(False)
        self.label_precision.setEnabled(False)
        self.spin_channel.setEnabled(False)
        self.check_stacked.setEnabled(False)
        self.label_channel.setEnabled(False)
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(False)
        self.button_remove_project.setEnabled(False)
        self.label_results.setEnabled(False)
        self.button_results_segmentation.setEnabled(False)
        self.button_results_morphometry.setEnabled(False)
        self.is_disabled = True

        self.slider_resource.setValue(0)
        self.slider_precision.setValue(0)

    def save_UI_state(self):
        return (self.slider_resource.isEnabled(),
                self.label_resource.isEnabled(),
                self.slider_precision.isEnabled(),
                self.label_precision.isEnabled(),
                self.spin_channel.isEnabled(),
                self.check_stacked.isEnabled(),
                self.label_channel.isEnabled(),
                self.button_results_segmentation.isEnabled(),
                self.button_start.isEnabled(),
                self.button_stop.isEnabled(),
                self.button_add_project.isEnabled(),
                self.button_remove_project.isEnabled(),
                self.slider_resource.value(),
                self.slider_precision.value(),
                self.spin_channel.value(),
                self.list_projects.isEnabled(),
                self.label_results.isEnabled(),
                self.button_results_segmentation.isEnabled(),
                self.button_results_morphometry.isEnabled(),
                )

    def restore_UI_state(self, _UI_state):
        self.is_loading = True
        self.slider_resource.setEnabled(_UI_state[0])
        self.label_resource.setEnabled(_UI_state[1])
        self.slider_precision.setEnabled(_UI_state[2])
        self.label_precision.setEnabled(_UI_state[3])
        self.spin_channel.setEnabled(_UI_state[4])
        self.check_stacked.setEnabled(_UI_state[5])
        self.label_channel.setEnabled(_UI_state[6])
        self.button_results_segmentation.setEnabled(_UI_state[7])
        self.button_start.setEnabled(_UI_state[8])
        self.button_stop.setEnabled(_UI_state[9])
        self.button_add_project.setEnabled(_UI_state[10])
        self.button_remove_project.setEnabled(_UI_state[11])
        self.slider_resource.setValue(_UI_state[12])
        self.slider_precision.setValue(_UI_state[13])
        self.spin_channel.setValue(_UI_state[14])
        self.list_projects.setEnabled(_UI_state[15])
        self.label_results.setEnabled(_UI_state[16]),
        self.button_results_segmentation.setEnabled(_UI_state[17]),
        self.button_results_morphometry.setEnabled(_UI_state[18]),
        self.is_loading = False

    def load_projects(self):
        project_dirs = [directory.name for directory in os.scandir(f"./{PROJECT_DIR}/") if directory.is_dir()]
        for directory in project_dirs:
            temp_item = self.list_projects.findItems(directory, Qt.MatchExactly)
            if len(temp_item) == 0:
                self.list_projects.addItem(directory)

        items = [self.list_projects.item(x) for x in range(self.list_projects.count())]
        for item in items:
            item_text = item.text()
            if item_text not in project_dirs:
                # Removing widgets from QListWidget is not working as intended
                # at the moment, so this is the way to go
                listItems = self.list_projects.selectedItems()
                if not listItems:
                    return
                self.list_projects.takeItem(self.list_projects.row(item))

    @staticmethod
    def save_project_configuration(_path, _configs):
        with open(_path, 'w+') as file:
            file.write(json.dumps(_configs))

    @staticmethod
    def load_project_configuration(_path):
        with open(_path, 'r') as file:
            return json.load(file)
