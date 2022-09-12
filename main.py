# Python Imports
import warnings
import logging
import sys

# Library Imports
from PySide6 import QtWidgets
from PySide6.QtGui import QFontDatabase

# Local Imports
from src.configs import LOG_START_APP_SIGNATURE, LOG_LEVEL
from src.ui.main_window import MainWindow
from src.utils import create_dark_palette, create_projects_dir

if __name__ == '__main__':
    # To silent Qt warnings
    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("amap.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(LOG_START_APP_SIGNATURE)
    logging.info("AMAP application Started")

    logging.debug("Creating projects directory")
    create_projects_dir()

    # Initializing Qt framework
    logging.debug("Creating Qt application object")
    app = QtWidgets.QApplication(sys.argv)

    logging.debug("Loading fonts")
    QFontDatabase.addApplicationFont("./res/fonts/Z003-MediumItalic.ttf")

    logging.debug("Creating the dark theme")
    QtWidgets.QApplication.setPalette(create_dark_palette())

    logging.debug("Creating the main window")
    window = MainWindow()
    window.show()

    logging.info("Executing the Qt app")
    sys.exit(app.exec())
