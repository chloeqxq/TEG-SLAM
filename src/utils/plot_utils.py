import os
import re

import cv2
import numpy as np
from PIL import Image


FRAME_NAME_PATTERN = re.compile(r"video_idx_(\d+)_kf_idx_(\d+)(?:_.+)?\.png$")


def _frame_sort_key(path):
    filename = os.path.basename(path)
    match = FRAME_NAME_PATTERN.match(filename)
    if match is not None:
        return (0, int(match.group(1)), int(match.group(2)), filename)

    trailing_number = re.search(r"(\d+)\.png$", filename)
    if trailing_number is not None:
        return (1, int(trailing_number.group(1)), 0, filename)

    return (2, 0, 0, filename)


def _list_png_files(directory_path, online=True):
    image_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".png")
    ]
    if online:
        image_files.sort(key=_frame_sort_key)
    else:
        image_files.sort()
    return image_files


def create_gif_from_directory(directory_path, output_filename, duration=100, online=True):
    """
    Creates a GIF from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the GIF.
    :param duration: Duration of each frame in the GIF (in milliseconds).
    """
    image_files = _list_png_files(directory_path, online=online)
    if not image_files:
        raise RuntimeError(f"No PNG files found in {directory_path}")

    # Load images
    images = [Image.open(file) for file in image_files]

    # Convert images to the same mode and size for consistency
    images = [img.convert('RGBA') for img in images]
    base_size = images[0].size
    resized_images = [img.resize(base_size, Image.LANCZOS) for img in images]

    # Save as GIF
    resized_images[0].save(output_filename, save_all=True, append_images=resized_images[1:], optimize=False, duration=duration, loop=0)


def create_video_from_directory(directory_path, output_filename, fps=10.0, online=True):
    """
    Creates an MP4 from all PNG images in a given directory.

    :param directory_path: Path to the directory containing PNG images.
    :param output_filename: Output filename for the MP4.
    :param fps: Frames per second for the output video.
    """
    image_files = _list_png_files(directory_path, online=online)
    if not image_files:
        raise RuntimeError(f"No PNG files found in {directory_path}")

    with Image.open(image_files[0]) as first_image:
        first_rgb = first_image.convert("RGB")
        base_size = first_rgb.size
        first_frame_bgr = cv2.cvtColor(np.array(first_rgb), cv2.COLOR_RGB2BGR)

    writer = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        base_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_filename}")

    try:
        writer.write(first_frame_bgr)
        for file in image_files[1:]:
            with Image.open(file) as image:
                image_rgb = image.convert("RGB")
                if image_rgb.size != base_size:
                    image_rgb = image_rgb.resize(base_size, Image.LANCZOS)
                frame_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
    finally:
        writer.release()