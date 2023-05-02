import os

import cv2
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def write_video(file_path, frames, fps, reversed=True):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames.reverse()

    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    writer.release()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def shrink_and_paste_on_blank(current_image, mask_width):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    """

    height = current_image.height
    width = current_image.width

    # shrink down by mask_width
    prev_image = current_image.resize((height - 2 * mask_width, width - 2 * mask_width))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)

    # create blank non-transparent image
    blank_image = np.array(current_image.convert("RGBA")) * 0
    blank_image[:, :, 3] = 1

    # paste shrinked onto blank
    blank_image[mask_width : height - mask_width, mask_width : width - mask_width, :] = prev_image
    prev_image = Image.fromarray(blank_image)

    return prev_image


def dummy(images, **kwargs):
    return images, False
