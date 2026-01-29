import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import cv2
import numpy as np
import csv
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ParticleAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dust Particle Analyzer')
        self.setGeometry(100, 100, 1400, 700)
        
        # Store current image path for re-analysis
        self.current_image_path = None
        self.batch_folder_path = None

        # Create sidebar (left panel) with buttons and results text
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)
        sidebar.setMaximumWidth(525)

        # Create buttons
        self.load_button = QPushButton('Load Image')
        self.load_button.setMaximumWidth(262)
        self.load_button.clicked.connect(self.load_image)
        sidebar_layout.addWidget(self.load_button)

        self.load_ref_button = QPushButton('Load Reference Image')
        self.load_ref_button.setMaximumWidth(262)
        self.load_ref_button.clicked.connect(self.load_reference_image)
        sidebar_layout.addWidget(self.load_ref_button)

        # Create text label for analysis results
        self.results_label = QLabel('Load an image to analyze dust particles')
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.results_label.setWordWrap(True)
        sidebar_layout.addWidget(self.results_label)

        # Create histogram canvases
        self.hist_min = FigureCanvas(Figure(figsize=(3.75, 2.7)))
        self.hist_max = FigureCanvas(Figure(figsize=(3.75, 2.7)))
        self.hist_median = FigureCanvas(Figure(figsize=(3.75, 2.7)))
        self.hist_mean = FigureCanvas(Figure(figsize=(3.75, 2.7)))

        # Create 2x2 grid layout for histograms
        hist_grid = QWidget()
        hist_layout = QVBoxLayout()
        hist_grid.setLayout(hist_layout)

        # First row: Min and Max
        hist_row1 = QHBoxLayout()
        hist_row1.addWidget(self.hist_min)
        hist_row1.addWidget(self.hist_max)
        hist_layout.addLayout(hist_row1)

        # Second row: Median and Mean
        hist_row2 = QHBoxLayout()
        hist_row2.addWidget(self.hist_median)
        hist_row2.addWidget(self.hist_mean)
        hist_layout.addLayout(hist_row2)

        sidebar_layout.addWidget(hist_grid)
        sidebar_layout.addStretch()

        # Create control panel above the image
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # Pixel to micron conversion
        control_layout.addWidget(QLabel('Pixels to Microns:'))
        self.micron_spinbox = QDoubleSpinBox()
        self.micron_spinbox.setMinimum(0.001)
        self.micron_spinbox.setSingleStep(0.1)
        self.micron_spinbox.setValue(1.0)
        self.micron_spinbox.setDecimals(3)
        self.micron_spinbox.setMaximumWidth(150)
        self.micron_spinbox.valueChanged.connect(self.on_parameters_changed)
        control_layout.addWidget(self.micron_spinbox)

        # Thresholding method selection
        control_layout.addWidget(QLabel('Threshold Method:'))
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems([
            'BINARY_INV',
            'BINARY',
            'TRUNC',
            'TOZERO',
            'TOZERO_INV',
            'OTSU'
        ])
        self.threshold_combo.setMaximumWidth(200)
        self.threshold_combo.currentTextChanged.connect(self.on_threshold_method_changed)
        control_layout.addWidget(self.threshold_combo)

        # Threshold value spinbox
        control_layout.addWidget(QLabel('Threshold Value:'))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(255)
        self.threshold_spinbox.setValue(100)
        self.threshold_spinbox.setSingleStep(1)
        self.threshold_spinbox.setMaximumWidth(150)
        self.threshold_spinbox.valueChanged.connect(self.on_parameters_changed)
        control_layout.addWidget(self.threshold_spinbox)
        control_layout.addStretch()

        # Create image display label (right panel)
        self.image_label = QLabel('No image loaded')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)

        # Create right panel with control panel and image
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_layout.addWidget(control_panel)
        right_layout.addWidget(self.image_label)

        # Create batch fitting panel below image
        batch_panel = QWidget()
        batch_layout = QHBoxLayout()
        batch_panel.setLayout(batch_layout)

        batch_layout.addWidget(QLabel('Batch Folder:'))
        self.batch_folder_line = QLineEdit()
        self.batch_folder_line.setReadOnly(True)
        self.batch_folder_line.setPlaceholderText('Select a folder to batch process images')
        batch_layout.addWidget(self.batch_folder_line)

        self.select_folder_button = QPushButton('Select Folder')
        self.select_folder_button.setMaximumWidth(200)
        self.select_folder_button.clicked.connect(self.select_batch_folder)
        batch_layout.addWidget(self.select_folder_button)

        self.batch_fit_button = QPushButton('Batch Fit')
        self.batch_fit_button.setMaximumWidth(150)
        self.batch_fit_button.clicked.connect(self.batch_fit_images)
        batch_layout.addWidget(self.batch_fit_button)
        batch_layout.addStretch()

        right_layout.addWidget(batch_panel)

        # Create main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(sidebar)
        main_layout.addWidget(right_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Option()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg);;All Files (*)', options=options)
        if file_name:
            self.current_image_path = file_name
            self.analyze_image()

    def load_reference_image(self):
        self.current_image_path = 'Demo/LocationC.png'
        self.analyze_image()

    def on_parameters_changed(self):
        """Called when any parameter changes - re-analyze current image"""
        if self.current_image_path:
            self.analyze_image()

    def on_threshold_method_changed(self):
        """Called when threshold method changes - update spinbox enabled state and re-analyze"""
        is_otsu = self.threshold_combo.currentText() == 'OTSU'
        self.threshold_spinbox.setEnabled(not is_otsu)
        self.on_parameters_changed()

    def get_threshold_method(self):
        """Get the threshold method based on combo box selection"""
        method_name = self.threshold_combo.currentText()
        method_map = {
            'BINARY_INV': cv2.THRESH_BINARY_INV,
            'BINARY': cv2.THRESH_BINARY,
            'TRUNC': cv2.THRESH_TRUNC,
            'TOZERO': cv2.THRESH_TOZERO,
            'TOZERO_INV': cv2.THRESH_TOZERO_INV,
            'OTSU': cv2.THRESH_OTSU
        }
        return method_map.get(method_name, cv2.THRESH_BINARY_INV)

    def analyze_image(self):
        if not self.current_image_path:
            return

        # Load the image
        image = cv2.imread(self.current_image_path)
        if image is None:
            self.results_label.setText('Error: Could not load image')
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get threshold parameters from controls
        threshold_method = self.get_threshold_method()
        threshold_value = self.threshold_spinbox.value()
        pixel_to_micron = self.micron_spinbox.value()

        # Thresholding to find dark particles
        if threshold_method == cv2.THRESH_OTSU:
            _, thresh_dark = cv2.threshold(gray, 0, 255, threshold_method | cv2.THRESH_BINARY_INV)
        else:
            _, thresh_dark = cv2.threshold(gray, threshold_value, 255, threshold_method)
        dark_particles = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Count dark particles
        dark_count = len(dark_particles)

        # Analyze light particles
        light_count = self.analyze_light_particles(image)

        # Calculate total particles and percentage
        total_particles = dark_count + light_count
        percent_dark = (dark_count / total_particles * 100) if total_particles > 0 else 0

        # Measure each dark particle with rotated bounding boxes
        particle_measurements = []
        result_image = image.copy()
        
        for contour in dark_particles:
            if len(contour) >= 3:  # At least 3 points to form a polygon
                side_lengths = []
                
                # Get the minimum area rectangle at different rotations
                for angle in range(0, 90, 5):  # 0 to 85 degrees in 5-degree increments
                    # Rotate the contour points
                    M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
                    rotated_contour = cv2.transform(contour, M)
                    
                    # Get axis-aligned bounding box for the rotated contour
                    x, y, w, h = cv2.boundingRect(rotated_contour)
                    
                    # Store both width and height as side lengths
                    side_lengths.extend([w, h])
                
                # Calculate statistics for this particle
                min_side = min(side_lengths) * pixel_to_micron
                max_side = max(side_lengths) * pixel_to_micron
                median_side = np.median(side_lengths) * pixel_to_micron
                mean_side = np.mean(side_lengths) * pixel_to_micron
                
                particle_measurements.append({
                    'min': min_side,
                    'max': max_side,
                    'median': median_side,
                    'mean': mean_side
                })
                
                # Draw the original contour on the result image
                cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

        # Create histograms if we have measurements
        if particle_measurements:
            all_mins = [p['min'] for p in particle_measurements]
            all_maxs = [p['max'] for p in particle_measurements]
            all_medians = [p['median'] for p in particle_measurements]
            all_means = [p['mean'] for p in particle_measurements]
            
            # Plot histograms
            self.plot_histogram(self.hist_min, all_mins, 'Min', 'blue')
            self.plot_histogram(self.hist_max, all_maxs, 'Max', 'red')
            self.plot_histogram(self.hist_median, all_medians, 'Median', 'green')
            self.plot_histogram(self.hist_mean, all_means, 'Mean', 'orange')

        # Save the annotated image temporarily
        temp_path = 'temp_analysis.png'
        cv2.imwrite(temp_path, result_image)

        # Update the results text in sidebar
        results_text = f"""Analysis Results:

Total Particles Detected: {total_particles}

Total Dark Particles: {dark_count}

Percent of Dark Particles: {percent_dark:.2f}%"""
        self.results_label.setText(results_text)

        # Update the image display
        self.image_label.setPixmap(QPixmap(temp_path).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))

    def analyze_light_particles(self, image):
        # Thresholding to find light particles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_light = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        light_particles = cv2.findContours(thresh_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Count light particles
        light_count = len(light_particles)
        return light_count

    def plot_histogram(self, canvas, data, title, color):
        """Plot a histogram on the given canvas"""
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Size (microns)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        canvas.figure.tight_layout()
        canvas.draw()

    def select_batch_folder(self):
        """Allow user to select a folder for batch processing"""
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder for Batch Processing')
        if folder:
            self.batch_folder_path = folder
            self.batch_folder_line.setText(folder)

    def batch_fit_images(self):
        """Batch process all images in the selected folder"""
        if not self.batch_folder_path:
            self.results_label.setText('Error: No folder selected for batch processing')
            return

        # Get current settings
        threshold_method = self.get_threshold_method()
        threshold_value = self.threshold_spinbox.value()
        pixel_to_micron = self.micron_spinbox.value()

        # Gather all image files
        import os
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(self.batch_folder_path) 
                      if f.lower().endswith(image_extensions)]

        if not image_files:
            self.results_label.setText('Error: No image files found in selected folder')
            return

        # Prepare CSV data
        csv_rows = []
        csv_header = ['Filename', 'Total Particles', 'Total Dark Particles', 'Min', 'Max', 'Median', 'Mean']

        # Process each image
        for image_file in image_files:
            image_path = os.path.join(self.batch_folder_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            if threshold_method == cv2.THRESH_OTSU:
                _, thresh_dark = cv2.threshold(gray, 0, 255, threshold_method | cv2.THRESH_BINARY_INV)
            else:
                _, thresh_dark = cv2.threshold(gray, threshold_value, 255, threshold_method)

            dark_particles = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            dark_count = len(dark_particles)

            # Analyze light particles
            _, thresh_light = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            light_particles = cv2.findContours(thresh_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            light_count = len(light_particles)
            total_particles = dark_count + light_count

            # Measure each dark particle
            for contour in dark_particles:
                if len(contour) >= 3:  # At least 3 points to form a polygon
                    side_lengths = []
                    for angle in range(0, 90, 5):
                        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
                        rotated_contour = cv2.transform(contour, M)
                        x, y, w, h = cv2.boundingRect(rotated_contour)
                        side_lengths.extend([w, h])

                    min_side = min(side_lengths) * pixel_to_micron
                    max_side = max(side_lengths) * pixel_to_micron
                    median_side = np.median(side_lengths) * pixel_to_micron
                    mean_side = np.mean(side_lengths) * pixel_to_micron

                    csv_rows.append([
                        image_file,
                        total_particles,
                        dark_count,
                        f'{min_side:.3f}',
                        f'{max_side:.3f}',
                        f'{median_side:.3f}',
                        f'{mean_side:.3f}'
                    ])

        # Write CSV file
        # Ask user where to save the CSV file
        csv_output_path, _ = QFileDialog.getSaveFileName(
            self, 
            'Save Batch Results As', 
            os.path.join(self.batch_folder_path, 'particle_analysis_results.csv'),
            'CSV Files (*.csv);;All Files (*)'
        )
        
        if not csv_output_path:
            self.results_label.setText('Batch processing cancelled - no save location selected')
            return

        try:
            with open(csv_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_header)
                writer.writerows(csv_rows)
            
            self.results_label.setText(f'Batch processing complete!\n\nProcessed {len(image_files)} images\nResults saved to:\n{os.path.basename(csv_output_path)}')
        except Exception as e:
            self.results_label.setText(f'Error writing CSV file: {str(e)}')

def get_dark_stylesheet():
    """Return a modern dark stylesheet for the application"""
    return """
    QMainWindow {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    QWidget {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    QPushButton {
        background-color: #0d47a1;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 6px 16px;
        font-weight: bold;
        font-size: 16px;
    }
    
    QPushButton:hover {
        background-color: #1565c0;
    }
    
    QPushButton:pressed {
        background-color: #0d3a8a;
    }
    
    QLabel {
        color: #ffffff;
        font-size: 15px;
    }
    
    QSpinBox, QDoubleSpinBox {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
        border-radius: 3px;
        padding: 4px;
        font-size: 15px;
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #0d47a1;
    }
    
    QComboBox {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
        border-radius: 3px;
        padding: 4px;
        font-size: 15px;
    }
    
    QComboBox:hover {
        border: 1px solid #0d47a1;
    }
    
    QComboBox::drop-down {
        border: none;
        background-color: transparent;
    }
    
    QComboBox::down-arrow {
        image: none;
    }
    
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #ffffff;
        selection-background-color: #0d47a1;
        border: 1px solid #404040;
        font-size: 15px;
    }
    
    QScrollBar:vertical {
        background-color: #1e1e1e;
        width: 10px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #404040;
        border-radius: 5px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #505050;
    }
    
    QScrollBar:horizontal {
        background-color: #1e1e1e;
        height: 10px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #404040;
        border-radius: 5px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #505050;
    }
    """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(get_dark_stylesheet())
    window = ParticleAnalyzer()
    window.show()
    sys.exit(app.exec())