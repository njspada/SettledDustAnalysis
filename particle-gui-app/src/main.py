import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk
from particle_detection import detect_particles

class ParticleGUI:
    def __init__(self, master):
        self.master = master
        master.title("Particle Identification")

        # Get screen width and set window size
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        master.geometry(f"{window_width}x{window_height}")

        # Set image display size to 60% of screen width
        self.display_width = int(screen_width * 0.6)
        self.display_height = int(screen_height * 0.6)

        self.label = Label(master, text="Select an image file:")
        self.label.pack()

        self.select_button = Button(master, text="Browse", command=self.load_image)
        self.select_button.pack()

        self.threshold_slider = Scale(master, from_=0, to=255, orient=HORIZONTAL, label="Threshold",
                                     command=self.update_image_with_particles)
        self.threshold_slider.pack()

        self.plot_button = Button(master, text="Plot Particles", command=self.plot_particles)
        self.plot_button.pack()

        self.image_label = Label(master)
        self.image_label.pack()

        self.image_path = None
        self.tk_image = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.label.config(text=f"Selected: {self.image_path}")
            self.update_image_with_particles(self.threshold_slider.get())

    def update_image_with_particles(self, threshold_value):
        if not self.image_path:
            return
        threshold_value = int(threshold_value)
        num_particles, image_with_particles = detect_particles(self.image_path, threshold_value)
        # Convert to PIL Image for Tkinter
        image_rgb = cv2.cvtColor(image_with_particles, cv2.COLOR_RGB2BGR)
        pil_image = Image.fromarray(image_rgb)
        pil_image = pil_image.resize((self.display_width, self.display_height))
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image)
        self.label.config(text=f"Particles: {num_particles} at Threshold: {threshold_value}")

    def plot_particles(self):
        if self.image_path:
            threshold_value = self.threshold_slider.get()
            num_particles, image_with_particles = detect_particles(self.image_path, int(threshold_value))
            plt.figure(figsize=(10, 6))
            plt.imshow(image_with_particles)
            plt.title(f"Number of Particles: {num_particles} at Threshold: {threshold_value}")
            plt.axis('off')
            plt.show()
        else:
            self.label.config(text="Please select an image file first.")

if __name__ == "__main__":
    root = Tk()
    particle_gui = ParticleGUI(root)
    root.mainloop()