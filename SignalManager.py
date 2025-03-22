from PyQt5.QtCore import pyqtSignal, QObject

class SignalManager(QObject):
    image_loaded = pyqtSignal()  # Global signal for image loading
    contour_requested = pyqtSignal()  # Signal for triggering active contour

# Create a single instance of SignalManager to use globally
global_signal_manager = SignalManager()
