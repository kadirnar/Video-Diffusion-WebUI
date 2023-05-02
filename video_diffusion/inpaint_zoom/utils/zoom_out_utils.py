import cv2
import numpy as np
from PIL import Image


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    writer.release()


def dummy(images, **kwargs):
    return images, False


def preprocess_image(current_image, steps, image_size):
    next_image = np.array(current_image.convert("RGBA")) * 0
    prev_image = current_image.resize((image_size - 2 * steps, image_size - 2 * steps))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)
    next_image[:, :, 3] = 1
    next_image[steps : image_size - steps, steps : image_size - steps, :] = prev_image
    prev_image = Image.fromarray(next_image)

    return prev_image


def preprocess_mask_image(current_image):
    mask_image = np.array(current_image)[:, :, 3]  # assume image has alpha mask (use .mode to check for "RGBA")
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")

    return current_image, mask_image
