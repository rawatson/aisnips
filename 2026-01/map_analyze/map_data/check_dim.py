from PIL import Image
import os

def check_image_size(filename):
    try:
        with Image.open(filename) as img:
            print(f"{filename}: {img.size}")
    except Exception as e:
        print(f"Error checking {filename}: {e}")

check_image_size("locations.png")
