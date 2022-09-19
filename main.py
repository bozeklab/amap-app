# Python Imports
import warnings
import logging
import sys
import os
import traceback

# Library Imports
from PySide6 import QtWidgets, QtGui
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QMessageBox

# Local Imports
from src.configs import LOG_START_APP_SIGNATURE, LOG_LEVEL
from src.ui.main_window import MainWindow
from src.utils import create_dark_palette, create_projects_dir


# We use this function to handle all unhandled exception and report the error to user
def handle_exception(exc_type, exc_value, exc_traceback):
    # KeyboardInterrupt is a special case.
    # We don't raise the error dialog when it occurs.
    if issubclass(exc_type, KeyboardInterrupt):
        return

    filename, line, _, _ = traceback.extract_tb(exc_traceback).pop()
    filename = os.path.basename(filename)
    error = "%s: %s" % (exc_type.__name__, exc_value)

    QMessageBox.critical(None, "Error",
                         "<html>A critical error has occured.<br/> "
                         + "<b>%s</b><br/><br/>" % error
                         + "It occurred at <b>line %d</b> of file <b>%s</b>.<br/>" % (line, filename)
                         + "</html>")

    logging.error("Closed due to an error. This is the full error report:")
    logging.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    sys.exit(1)


# install handler for exceptions
sys.excepthook = handle_exception

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

    if sys.platform == 'linux':
        logging.debug("Creating the dark theme")
        QtWidgets.QApplication.setPalette(create_dark_palette())

    logging.debug("Creating the main window")
    window = MainWindow()
    window.show()

    logging.info("Executing the Qt app")
    sys.exit(app.exec())


