from tkinter import Tk, Label, Button, Scale, HORIZONTAL, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import cv2
from particle_detection import identify_particles

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Particle Identification GUI")

        self.label = Label(master, text="Select an image and adjust the threshold")
        self.label.pack()

        self.load_button = Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.threshold_scale = Scale(master, from_=0, to=255, orient=HORIZONTAL, label="Threshold")
        self.threshold_scale.pack()
        self.threshold_scale.bind("<Motion>", self.update_plot)

        self.canvas = Canvas(master, width=800, height=600)
        self.canvas.pack()

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.canvas)
        self.canvas_plot.get_tk_widget().pack()

        self.image = None
        self.particle_count = 0

    def load_image(self):
        # Load an image file
        image_path = 'C:/git/SettledDustAnalysis/Demo/LocationC.png'  # Placeholder for file dialog
        self.image = cv2.imread(image_path)
        self.update_plot()

    def update_plot(self, event=None):
        if self.image is not None:
            threshold_value = self.threshold_scale.get()
            self.particle_count = identify_particles(self.image, threshold_value)

            self.ax.clear()
            self.ax.plot(threshold_value, self.particle_count, 'ro')
            self.ax.set_xlabel("Threshold Value")
            self.ax.set_ylabel("Number of Particles")
            self.ax.set_title("Threshold Sensitivity vs Number of Particles")
            self.canvas_plot.draw()

def main():
    root = Tk()
    gui = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()