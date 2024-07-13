import matplotlib.pyplot as plt
import numpy as np

print("Import image_processing.py...")

def save_PIL_image(img, filename):
    img = np.array(img.convert("RGB"))
    plt.imsave(f"./images/{filename}", img)