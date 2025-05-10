#!/usr/bin/env python3

import os
import glob
import argparse
from PIL import Image


def create_gif(input_path, output_path, duration=100, loop=0):
    """
    Create a GIF from a sequence of images
    
    Parameters:
    - input_path: Directory containing images or a pattern like "path/to/images/*.png"
    - output_path: Path where the GIF will be saved
    - duration: Duration of each frame in milliseconds
    - loop: Number of times to loop the GIF (0 = infinite)
    """
    print(f"Creating GIF from images in {input_path}")
    
    # Get list of image files
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.*")))
        # Filter for common image extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in files if f.lower().endswith(image_extensions)]
    else:
        files = sorted(glob.glob(input_path))
    
    if not files:
        print(f"No image files found at {input_path}")
        return
    
    print(f"Found {len(files)} image files")
    
    # Open all images
    images = []
    for file in files:
        try:
            img = Image.open(file)
            images.append(img.copy())
            print(f"Added: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not images:
        print("No valid images found!")
        return
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    print(f"GIF created successfully at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from a sequence of images")
    parser.add_argument("input", help="Directory containing images or a pattern like 'path/to/images/*.png'")
    parser.add_argument("-o", "--output", default="output.gif", help="Output GIF filename")
    parser.add_argument("-d", "--duration", type=int, default=1500, help="Duration of each frame in milliseconds")
    parser.add_argument("-l", "--loop", type=int, default=0, help="Number of times to loop (0 = infinite)")
    
    args = parser.parse_args()
    create_gif(args.input, args.output, args.duration, args.loop)