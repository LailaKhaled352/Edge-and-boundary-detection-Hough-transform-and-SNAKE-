from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QEvent, Qt
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from SignalManager import global_signal_manager  

class ImageViewer(QWidget):
    def __init__(self, input_view=None, output_view=None, index=0, img_num=None,mode=False, widget=0):
        super().__init__()
        self.img_num = img_num
        self.index = index
        self._image_path = None
        self.img_data = None
        self._is_grey = False
        self.input_view = input_view
        self.output_view = output_view

        self.mode = mode #to display colored image for second tab
        self.widget = widget  # Determines if drawing is enabled
         # Variables for rectangle drawing
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rect_label = None
        self.setup_double_click_event()
        if self.widget == 2 and self.input_view:
            print("enter")
            self.input_view.setMouseTracking(True)
            self.input_view.installEventFilter(self)
       

       
    
    def setup_double_click_event(self):
        if self.input_view:
            self.input_view.mouseDoubleClickEvent = self.handle_double_click
    
    def eventFilter(self, obj, event):
        if obj == self.input_view:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.RightButton:
                    self.drawing = True
                    self.start_point = event.pos()
                    self.end_point = event.pos()
                    self.update_rectangle()
                    return True

            elif event.type() == QEvent.MouseMove and self.drawing:
                self.end_point = event.pos()
                self.update_rectangle()
                return True

            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.RightButton and self.drawing:
                    self.drawing = False
                    global_signal_manager.contour_requested.emit()  # Emit signal to start active contour
                    return True

            elif event.type() == QEvent.MouseButtonDblClick:
                self.handle_double_click()
                return True

        return super().eventFilter(obj, event)

    



    def update_rectangle(self):
        """ Draw a rectangle dynamically on the image view """
        if not self.start_point or not self.end_point:
            return

        x1, y1 = self.start_point.x(), self.start_point.y()
        x2, y2 = self.end_point.x(), self.end_point.y()
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
        if self.rect_label:
         self.rect_label.deleteLater()
         self.rect_label = None  

        self.rect_label = QLabel(self.input_view)
        self.rect_label.setStyleSheet("border: 2px solid red; background: rgba(255, 0, 0, 50);")
        self.rect_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.rect_label.setGeometry(x, y, w, h)
        self.rect_label.show()

    def handle_double_click(self, event=None):
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.browse_image(file_path)
    
    def browse_image(self, image_path):
        self._image_path = image_path
        if self.check_extension():
         if self.mode:
                print("enter333")
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
                self._is_grey = False
         else:   

            if self.index in [0, 2]:
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_GRAYSCALE)
                self._is_grey = True
            else:
                self.img_data = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
                self._is_grey = False

         if self.img_data is None:
                print("Error loading image.")
                return

            

        if self.output_view:
             for child in self.output_view.findChildren(QLabel):
              child.deleteLater() 




        if self.rect_label:
             self.rect_label.deleteLater()
             self.rect_label = None  # Reset 


        self._processed_image = self.img_data  # Store the processed image


        if self.input_view:
                if self.index==0:
                        self.display_image(self.img_data, self.input_view)
                elif self.index==1:
                        self.display_RGB_image(self.img_data, self.input_view)
                elif self.index==2:
                    self.display_image(self.img_data, self.input_view)

            
    
    def display_output_image(self, processed_img=None, output=None):
        if processed_img is None:
            processed_img = self._processed_image  
        if processed_img is None:
            print("No processed image to display.")
            return
        if output:
            self.display_image(processed_img, output)
        elif self.output_view:
            self.display_image(processed_img, self.output_view)
    
    
    def display_image(self, img, target):
        
        if img is None or not isinstance(img, np.ndarray):
            print("Invalid image data.")
            return


         # Determine image mode and convert accordingly
        if self.mode:  # Color mode
         print ("enter2")
         if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

         else:
            print("Expected a color image but got grayscale.")
            return
        else:  # Grayscale mode
         if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         q_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
       

       

        if q_image.isNull():
            print("Failed to create QImage.")
            return

       
        pixmap = QPixmap.fromImage(q_image)

        
        for child in target.findChildren(QLabel):
            child.deleteLater()

        label = QLabel(target)
        label.setPixmap(pixmap.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        label.setScaledContents(True)
        label.setGeometry(0, 0, target.width(), target.height())

        # Ensure the label is visible and on top
        label.show()
        label.raise_()

        print(f"{'Color' if self.mode else 'Grayscale'} image displayed in widget with size: {target.size() , self.input_view }")
        if self.mode:
            global_signal_manager.image_loaded.emit()  # Emit the global signal



    def display_RGB_image(self, img, target):
        print("1")
        if img is None or not isinstance(img, np.ndarray):
            print("Invalid image data.")
            return

       
        height, width, channels = img.shape
        bytes_per_line = 3 * width  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.transformation_cls.set_colored_img(img)
        gray_img=self.transformation_cls.convert_to_grayscale()
        self.display_output_image(gray_img)
        self.transformation_cls.plot_histograms()

        
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        if q_image.isNull():
            print("Failed to create QImage.")
            return

        
        pixmap = QPixmap.fromImage(q_image)

        
        for child in target.findChildren(QLabel):
            child.deleteLater()
        label = QLabel(target)
        label.setPixmap(pixmap.scaled(target.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        label.setScaledContents(True)
        label.setGeometry(0, 0, target.width(), target.height())

        # Ensure the label is visible and on top
        label.show()
        label.raise_()

        print(f"RGB image displayed in widget with size: {target.size()}")      

    def check_extension(self):
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        if not any(self._image_path.lower().endswith(ext) for ext in valid_extensions):
            print("Invalid image file extension.")
            self._image_path = None  
            return False
        print("Valid image file extension.")
        return True
    

    def apply_filtered_image(self, filtered_img, freq=None):
        """ Update the input view if a frequency string is provided, otherwise use the output view. """
        if filtered_img is not None:
            if freq:  
                self.display_image(filtered_img, self.input_view)
            else:
                
                self.display_image(filtered_img, self.output_view)
    
    def check_grey_scale(self, image):
        if len(image.shape) == 2:
            return True  
        b, g, r = cv2.split(image)
        return cv2.countNonZero(b - g) == 0 and cv2.countNonZero(b - r) == 0
    
    def get_loaded_image(self):
        return self.img_data


    def display_contour(self, contour):
     """
     Displays the active contour on top of the original image in the output widget.
    
     :param contour: NumPy array of contour points (Nx2)
      """
     if self.img_data is None:
        print("No image loaded.")
        return
    # Clear previous contour before drawing a new one
     self._processed_image = self.img_data.copy()

    # Draw the contour (RED line)
     for i in range(len(contour) - 1):
        pt1 = tuple(contour[i].astype(int))
        pt2 = tuple(contour[i + 1].astype(int))
        cv2.line(self._processed_image, pt1, pt2, (0, 0, 255), 4)  # Red line with thickness 4

    # Connect last and first point to close the contour
     if len(contour) > 1:
        cv2.line(self._processed_image, tuple(contour[-1].astype(int)), tuple(contour[0].astype(int)), (0, 0, 255), 4)
   

     contour_image = cv2.resize(self._processed_image, (self.img_data.shape[1], self.img_data.shape[0]))

    # Display the result in the output widget
     self.display_output_image(contour_image)

    
    
    def get_initial_contour(self):
     """Get the rectangle coordinates (x1, y1, x2, y2) for use as an initial contour."""
     if self.start_point and self.end_point:
        x1, y1 = self.start_point.x(), self.start_point.y()
        x2, y2 = self.end_point.x(), self.end_point.y()
        
        # Ensure the order is always correct
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        return x1, y1, x2, y2  
     return None  # Return None if no rectangle is selected