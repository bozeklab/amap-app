# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QSpinBox, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1023, 690)
        font = QFont()
        font.setFamilies([u"Times New Roman"])
        MainWindow.setFont(font)
        self.actionNew_Project = QAction(MainWindow)
        self.actionNew_Project.setObjectName(u"actionNew_Project")
        self.actionNew_Project.setFont(font)
        self.actionSettings = QAction(MainWindow)
        self.actionSettings.setObjectName(u"actionSettings")
        self.actionSettings.setFont(font)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionExit.setFont(font)
        self.actionHow_to = QAction(MainWindow)
        self.actionHow_to.setObjectName(u"actionHow_to")
        self.actionSource = QAction(MainWindow)
        self.actionSource.setObjectName(u"actionSource")
        self.actionAbout_us = QAction(MainWindow)
        self.actionAbout_us.setObjectName(u"actionAbout_us")
        self.widget_central = QWidget(MainWindow)
        self.widget_central.setObjectName(u"widget_central")
        self.verticalLayout = QVBoxLayout(self.widget_central)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.top_frame = QFrame(self.widget_central)
        self.top_frame.setObjectName(u"top_frame")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.top_frame.sizePolicy().hasHeightForWidth())
        self.top_frame.setSizePolicy(sizePolicy)
        self.top_frame.setFrameShape(QFrame.Shape.NoFrame)
        self.top_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.top_frame.setLineWidth(0)
        self.top_frame.setMidLineWidth(1)
        self.gridLayout = QGridLayout(self.top_frame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_header = QLabel(self.top_frame)
        self.label_header.setObjectName(u"label_header")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.label_header.sizePolicy().hasHeightForWidth())
        self.label_header.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setFamilies([u"Z003"])
        font1.setPointSize(48)
        font1.setBold(False)
        font1.setItalic(True)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.label_header.setFont(font1)
        self.label_header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.label_header.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.label_header.setAutoFillBackground(False)
        self.label_header.setFrameShadow(QFrame.Shadow.Plain)
        self.label_header.setScaledContents(False)
        self.label_header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_header, 0, 0, 1, 1)

        self.label_image = QLabel(self.top_frame)
        self.label_image.setObjectName(u"label_image")
        sizePolicy2 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(2)
        sizePolicy2.setHeightForWidth(self.label_image.sizePolicy().hasHeightForWidth())
        self.label_image.setSizePolicy(sizePolicy2)
        self.label_image.setFrameShape(QFrame.Shape.Panel)
        self.label_image.setFrameShadow(QFrame.Shadow.Plain)
        self.label_image.setPixmap(QPixmap(u"../../res/images/header.png"))
        self.label_image.setScaledContents(True)

        self.gridLayout.addWidget(self.label_image, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.top_frame)

        self.frame_buttom = QFrame(self.widget_central)
        self.frame_buttom.setObjectName(u"frame_buttom")
        self.frame_buttom.setEnabled(True)
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(3)
        sizePolicy3.setHeightForWidth(self.frame_buttom.sizePolicy().hasHeightForWidth())
        self.frame_buttom.setSizePolicy(sizePolicy3)
        self.frame_buttom.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_buttom.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame_buttom)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.vertical_layout_projects = QVBoxLayout()
        self.vertical_layout_projects.setObjectName(u"vertical_layout_projects")
        self.label_projects = QLabel(self.frame_buttom)
        self.label_projects.setObjectName(u"label_projects")

        self.vertical_layout_projects.addWidget(self.label_projects)

        self.list_widget_projects = QListWidget(self.frame_buttom)
        self.list_widget_projects.setObjectName(u"list_widget_projects")

        self.vertical_layout_projects.addWidget(self.list_widget_projects)

        self.horizontal_remove_add = QHBoxLayout()
        self.horizontal_remove_add.setObjectName(u"horizontal_remove_add")
        self.button_remove_project = QPushButton(self.frame_buttom)
        self.button_remove_project.setObjectName(u"button_remove_project")
        self.button_remove_project.setEnabled(False)

        self.horizontal_remove_add.addWidget(self.button_remove_project)

        self.button_add_project = QPushButton(self.frame_buttom)
        self.button_add_project.setObjectName(u"button_add_project")

        self.horizontal_remove_add.addWidget(self.button_add_project)


        self.vertical_layout_projects.addLayout(self.horizontal_remove_add)


        self.horizontalLayout.addLayout(self.vertical_layout_projects)

        self.frame_projects = QFrame(self.frame_buttom)
        self.frame_projects.setObjectName(u"frame_projects")
        sizePolicy4 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.frame_projects.sizePolicy().hasHeightForWidth())
        self.frame_projects.setSizePolicy(sizePolicy4)
        self.frame_projects.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_projects.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_projects)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.grid_project = QGridLayout()
        self.grid_project.setSpacing(6)
        self.grid_project.setObjectName(u"grid_project")
        self.label_mem_allocation = QLabel(self.frame_projects)
        self.label_mem_allocation.setObjectName(u"label_mem_allocation")
        self.label_mem_allocation.setEnabled(False)

        self.grid_project.addWidget(self.label_mem_allocation, 2, 0, 1, 1)

        self.horizontal_result = QHBoxLayout()
        self.horizontal_result.setObjectName(u"horizontal_result")
        self.label_results = QLabel(self.frame_projects)
        self.label_results.setObjectName(u"label_results")
        self.label_results.setEnabled(False)
        self.label_results.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.horizontal_result.addWidget(self.label_results)

        self.button_result_segmentation = QPushButton(self.frame_projects)
        self.button_result_segmentation.setObjectName(u"button_result_segmentation")
        self.button_result_segmentation.setEnabled(False)

        self.horizontal_result.addWidget(self.button_result_segmentation)

        self.button_result_morphometry = QPushButton(self.frame_projects)
        self.button_result_morphometry.setObjectName(u"button_result_morphometry")
        self.button_result_morphometry.setEnabled(False)

        self.horizontal_result.addWidget(self.button_result_morphometry)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontal_result.addItem(self.horizontalSpacer_2)


        self.grid_project.addLayout(self.horizontal_result, 3, 0, 1, 5)

        self.label_cpu_allocation = QLabel(self.frame_projects)
        self.label_cpu_allocation.setObjectName(u"label_cpu_allocation")
        self.label_cpu_allocation.setEnabled(False)

        self.grid_project.addWidget(self.label_cpu_allocation, 0, 0, 1, 1)

        self.slider_mem_allocation = QSlider(self.frame_projects)
        self.slider_mem_allocation.setObjectName(u"slider_mem_allocation")
        self.slider_mem_allocation.setEnabled(False)
        self.slider_mem_allocation.setMaximum(4)
        self.slider_mem_allocation.setValue(0)
        self.slider_mem_allocation.setSliderPosition(0)
        self.slider_mem_allocation.setOrientation(Qt.Orientation.Horizontal)
        self.slider_mem_allocation.setTickPosition(QSlider.TickPosition.TicksBothSides)

        self.grid_project.addWidget(self.slider_mem_allocation, 2, 1, 1, 4)

        self.slider_cpu_allocation = QSlider(self.frame_projects)
        self.slider_cpu_allocation.setObjectName(u"slider_cpu_allocation")
        self.slider_cpu_allocation.setEnabled(False)
        self.slider_cpu_allocation.setMaximum(4)
        self.slider_cpu_allocation.setValue(0)
        self.slider_cpu_allocation.setSliderPosition(0)
        self.slider_cpu_allocation.setOrientation(Qt.Orientation.Horizontal)
        self.slider_cpu_allocation.setTickPosition(QSlider.TickPosition.TicksBothSides)

        self.grid_project.addWidget(self.slider_cpu_allocation, 0, 1, 1, 4)

        self.grid_configs = QGridLayout()
        self.grid_configs.setObjectName(u"grid_configs")
        self.check_stacked = QCheckBox(self.frame_projects)
        self.check_stacked.setObjectName(u"check_stacked")
        self.check_stacked.setEnabled(False)

        self.grid_configs.addWidget(self.check_stacked, 1, 0, 1, 1)

        self.spin_channel = QSpinBox(self.frame_projects)
        self.spin_channel.setObjectName(u"spin_channel")
        self.spin_channel.setEnabled(False)
        self.spin_channel.setMaximum(20)

        self.grid_configs.addWidget(self.spin_channel, 0, 1, 1, 1)

        self.label_channel = QLabel(self.frame_projects)
        self.label_channel.setObjectName(u"label_channel")
        self.label_channel.setEnabled(False)

        self.grid_configs.addWidget(self.label_channel, 0, 0, 1, 1)

        self.check_old_roi = QCheckBox(self.frame_projects)
        self.check_old_roi.setObjectName(u"check_old_roi")
        self.check_old_roi.setEnabled(False)

        self.grid_configs.addWidget(self.check_old_roi, 2, 0, 1, 1)


        self.grid_project.addLayout(self.grid_configs, 4, 0, 1, 2)

        self.horizontalSpacer = QSpacerItem(30, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Minimum)

        self.grid_project.addItem(self.horizontalSpacer, 4, 2, 1, 1)

        self.horizontal_stop_start = QHBoxLayout()
        self.horizontal_stop_start.setObjectName(u"horizontal_stop_start")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontal_stop_start.addItem(self.horizontalSpacer_3)

        self.button_stop = QPushButton(self.frame_projects)
        self.button_stop.setObjectName(u"button_stop")
        self.button_stop.setEnabled(False)

        self.horizontal_stop_start.addWidget(self.button_stop)

        self.button_start = QPushButton(self.frame_projects)
        self.button_start.setObjectName(u"button_start")
        self.button_start.setEnabled(False)

        self.horizontal_stop_start.addWidget(self.button_start)


        self.grid_project.addLayout(self.horizontal_stop_start, 5, 2, 1, 1)


        self.horizontalLayout_4.addLayout(self.grid_project)


        self.horizontalLayout.addWidget(self.frame_projects)


        self.verticalLayout.addWidget(self.frame_buttom)

        MainWindow.setCentralWidget(self.widget_central)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionNew_Project.setText(QCoreApplication.translate("MainWindow", u"New project", None))
        self.actionSettings.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionHow_to.setText(QCoreApplication.translate("MainWindow", u"How to?", None))
        self.actionSource.setText(QCoreApplication.translate("MainWindow", u"GitHub", None))
        self.actionAbout_us.setText(QCoreApplication.translate("MainWindow", u"About us", None))
        self.label_header.setText(QCoreApplication.translate("MainWindow", u"Welcome to AMAP application", None))
        self.label_image.setText("")
        self.label_projects.setText(QCoreApplication.translate("MainWindow", u"Projects:", None))
        self.button_remove_project.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        self.button_add_project.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.label_mem_allocation.setText(QCoreApplication.translate("MainWindow", u"Memory allocation:", None))
        self.label_results.setText(QCoreApplication.translate("MainWindow", u"Results:", None))
        self.button_result_segmentation.setText(QCoreApplication.translate("MainWindow", u"Segmentation", None))
        self.button_result_morphometry.setText(QCoreApplication.translate("MainWindow", u"Morphometry", None))
        self.label_cpu_allocation.setText(QCoreApplication.translate("MainWindow", u"CPU allocation:", None))
        self.check_stacked.setText(QCoreApplication.translate("MainWindow", u"Stacked", None))
        self.label_channel.setText(QCoreApplication.translate("MainWindow", u"Target channel", None))
        self.check_old_roi.setText(QCoreApplication.translate("MainWindow", u"Old ROI algorithm (AMAP)", None))
        self.button_stop.setText(QCoreApplication.translate("MainWindow", u"Stop", None))
        self.button_start.setText(QCoreApplication.translate("MainWindow", u"Start", None))
    # retranslateUi

