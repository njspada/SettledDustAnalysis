# Particle Detection GUI Application

This project is a graphical user interface (GUI) application for detecting particles in images. Users can select an image file, adjust the threshold sensitivity using a slider, and visualize the number of detected particles based on the selected threshold.

## Project Structure

```
particle-gui-app
├── src
│   ├── main.py               # Entry point of the application
│   ├── gui.py                # Contains the GUI class and components
│   ├── particle_detection.py  # Functions for particle detection
│   └── utils.py              # Utility functions for image processing
├── requirements.txt          # List of dependencies
└── README.md                 # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd particle-gui-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

Once the application is running, you can:

- Select an image file using the file dialog.
- Adjust the threshold sensitivity using the slider.
- View the plot showing the number of detected particles based on the selected threshold.

## Dependencies

The project requires the following Python libraries:

- OpenCV
- Matplotlib
- Tkinter
- NumPy
- SciPy
- scikit-image

Make sure to install these libraries using the `requirements.txt` file provided.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.