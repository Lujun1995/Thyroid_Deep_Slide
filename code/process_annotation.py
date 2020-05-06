"""
PTC Whole Slide
Contains all functions for processing annotation images and producing corresponding labels.



Modify is_purple to is_white
Delete BALANCING CLASS DISTRIBUTION (using different weights in the model instead)

"""

import functools
import itertools
import math
import time
import os
from multiprocessing import (Process, Queue, RawArray)
from pathlib import Path
from shutil import copyfile
from typing import (Callable, Dict, List, Tuple)

import numpy as np
from PIL import Image
from imageio import (imsave, imread)
from skimage.measure import block_reduce

from utils import (get_all_image_paths, get_image_names, get_image_paths,
                   get_subfolder_paths)

Image.MAX_IMAGE_PIXELS = None

###########################################
#         FINDING COLOR AREA              #
###########################################
def is_red(crop: np.ndarray, threshold: int,
           scale_size: int) -> bool:
    """
    Determines if a given portion of an image is annotated.
    Args:
        crop: Portion of the image to check for being annotated.
        threshold: Number of points for region to be considered annotation.
        scale_size: Scalar to use for reducing image to check.
    """
    block_size = (crop.shape[0] // scale_size,
                  crop.shape[1] // scale_size, 1)

    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > 200
    cond2 = g < 100
    cond3 = b < 100
    pooled = pooled[cond1 & cond2 & cond3]
    num_color = pooled.shape[0]

    return num_color > threshold



def is_white(crop: np.ndarray, white_threshold: int,
             scale_size: int) -> bool:
    """
    Determines if a given portion of an image is white.
    Args:
        crop: Portion of the image to check for being white.
        white_threshold: Number of white points for region to be considered white.
        scale_size: Scalar to use for reducing image to check for white.
    Returns:
        A boolean representing whether the image is white or not.
    """
    block_size = (crop.shape[0] // scale_size,
                  crop.shape[1] // scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is white.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > 200
    cond2 = b > 200
    cond3 = g > 200
    cond4 = abs(r-b) < 10
    cond5 = abs(g-b) < 10
    cond6 = abs(r-g) < 10

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
    num_white = pooled.shape[0]

    return num_white > white_threshold


###########################################
#     SAVING LOCATION OF COLOR AREAS      #
###########################################
def label_patch(xy_start: Tuple[int, int], image: np.ndarray,
                patch_size: int, white_threshold: int, threshold: int, scale_size: int) -> int:
    """
    Determines if a patch is annotated.
    Args:
        crop: Portion of the image to check for being annotated.
        threshold: Number of points for region to be considered annotation.
        scale_size: Scalar to use for reducing image to check.
    """

    x_start, y_start = xy_start

    patch = image[x_start:x_start + patch_size, y_start:y_start +
                  patch_size, :]
    patch = patch[..., [0, 1, 2]]

    result = None

    if not is_white(crop=patch, white_threshold=white_threshold, scale_size=scale_size):
        result = 'Benign'
        if is_red(crop=patch, threshold=threshold, scale_size=scale_size):
            result = 'PTC'
    return result


def find_patches_location(input_folder: Path, output_folder: Path, by_folder: bool,
                          patch_size: int, white_threshold: int, threshold: int,
                          scale_size: int) -> None:

    output_folder.mkdir(parents=True, exist_ok=True)
    image_locs = get_all_image_paths(
        master_folder=input_folder) if by_folder else get_image_names(
            folder=input_folder)

    for image_loc in image_locs:
        image = imread(
            uri=(image_loc if by_folder else input_folder.joinpath(image_loc)), pilmode="RGB")

        img = RawArray(
            typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
            size_or_initializer=image.size)
        img_np = np.frombuffer(buffer=img,
                               dtype=image.dtype).reshape(image.shape)
        np.copyto(dst=img_np, src=image)

        # Number of x starting points.
        x_steps = int((image.shape[0] - patch_size) / patch_size) + 1
        # Number of y starting points.
        y_steps = int((image.shape[1] - patch_size) / patch_size) + 1
        # Step size, same for x and y.
        step_size = int(patch_size)

        filename = image_loc.name.split(".")[0]

        with output_folder.joinpath(f"{filename}.csv").open(mode="w") as writer:
            writer.write("x_start,y_start,label\n")
            for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                        range(0, y_steps * step_size, step_size)):

                result = label_patch(xy_start=xy, image=img_np,
                                     patch_size=patch_size, white_threshold=white_threshold,
                                     threshold=threshold, scale_size=scale_size)
                if result is not None:
                    writer.write(f"{','.join([str(xy[0]), str(xy[1]), result])}\n")





def add_labels_to_image(xy_to_label_class: Dict[Tuple[str, str], str],
                        image: np.ndarray, label_color: Dict[str, np.array],
                        patch_size: int, produce_patches: bool, output_folder: Path,
                        image_ext: str, image_name: str, border: int) -> np.ndarray:
    """
    Overlay the color on the WSI.
    Args:
        xy_to_label_class: Dictionary mapping coordinates to predicted class
        along with the confidence.
        image: Image to add color and be cropped.
        label_color: Dictionary mapping string color to NumPy ndarray color.
        patch_size: Size of the patches extracted from the image.
        produce_patches: Whether to produce patches.
        output_folder: The output folder for cropped patche.
        image_ext: Image extension.
        image_loc: The location of the input image.
        border: The length of the border around the labeled patches.
    Returns:
        The image with highlight in labeled area.
    """

    for x_start, y_start in xy_to_label_class.keys():
        label = xy_to_label_class[x_start, y_start]
        output_subfolder = output_folder.joinpath(f"{label}")
        output_subfolder.mkdir(parents=True, exist_ok=True)
        x_start = int(x_start)
        y_start = int(y_start)

        image_name = image_name.split(".")[0]
        start = 0
        end = patch_size
        if produce_patches:
            output_path = output_subfolder.joinpath(f"{image_name}_{x_start}_{y_start}.{image_ext}")
            patch = image[x_start + start:x_start + end, y_start + start:y_start + end, :]
            imsave(uri=output_path, im=patch)
        image[x_start + start:x_start + end, y_start + start:y_start + end, :] = label_color[label]*0.5 + image[x_start + start:x_start + end, y_start + start:y_start +end, :]*0.5
        image[x_start + start:x_start + end, y_start + start:y_start + border, :] = np.array([0, 0, 0])
        image[x_start + start:x_start + end, y_start + end - border:y_start + end, :] = np.array([0, 0, 0])
        image[x_start + start:x_start + start + border, y_start + start:y_start + end, :] = np.array([0, 0, 0])
        image[x_start + end - border:x_start + end, y_start + start:y_start + end, :] = np.array([0, 0, 0])
    return image


def get_xy_to_label_class(labels_folder: Path, img_name: str
                         ) -> Dict[Tuple[str, str], str]:
    """
    Find the dictionary of predictions.
    Args:
        window_prediction_folder: Path to the folder containing a CSV file with the predicted classes.
        img_name: Name of the image to find the predicted classes for.
    Returns:
        A dictionary mapping image coordinates to the predicted class and the confidence of the prediction.
    """
    xy_to_pred_class = {}

    with labels_folder.joinpath(img_name).with_suffix(".csv").open(
            mode="r") as csv_lines_open:
        csv_lines = csv_lines_open.readlines()[1:]

        predictions = [line[:-1].split(",") for line in csv_lines]
        for prediction in predictions:
            x = prediction[0]
            y = prediction[1]
            label = prediction[2]
            # Implement thresholding.
            xy_to_pred_class[(x, y)] = label
    return xy_to_pred_class







def vis_produce_patches(image_folder: Path, labels_folder: Path, vis_folder: Path,
                        patch_folder: Path, produce_patches: bool,
                        image_ext: str, label_color: Dict[str, np.array], patch_size: int,
                        border: int, label: List[str]) -> None:
    """
    Main function for visualization.
    Args:
        wsi_folder: Path to WSI.
        preds_folder: Path containing the predicted classes.
        vis_folder: Path to output the WSI with overlaid classes to.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        colors: Colors to use for visualization.
        patch_size: Size of the patches extracted from the WSI.
    """
    # Find list of WSI.
    whole_slides = get_all_image_paths(master_folder=image_folder)
    print(f"{len(whole_slides)} whole slides found from {image_folder}")

    patch_folder.mkdir(parents=True, exist_ok=True)

    # Go over all of the WSI.
    for whole_slide in whole_slides:
        # Read in the image.
        whole_slide_numpy = imread(uri=whole_slide)[..., [0, 1, 2]]


        assert whole_slide_numpy.shape[
            2] == 3, f"Expected 3 channels while your image has {whole_slide_numpy.shape[2]} channels."

        # Save it.
        output_path = Path(
            f"{vis_folder.joinpath(whole_slide.name).with_suffix('')}"
            f"_labeled.jpg")

        # Confirm the output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temporary fix. Need not to make folders with no crops.

        xy_to_label_class = get_xy_to_label_class(labels_folder=labels_folder, img_name=whole_slide.name)
        image = add_labels_to_image(xy_to_label_class=xy_to_label_class, image=whole_slide_numpy,
                                    label_color=label_color, patch_size=patch_size, produce_patches=produce_patches,
                                    output_folder=patch_folder, image_ext=image_ext, image_name=whole_slide.name, border=border)

        imsave(uri=output_path, im=image)

    label_1_num = len([name for name in os.listdir(patch_folder.joinpath(f"{label[0]}"))])
    label_2_num = len([name for name in os.listdir(patch_folder.joinpath(f"{label[1]}"))])

    print(f"find the image with labels in {vis_folder}")
    print(f"produce patches in {patch_folder}")
    print(f"{label_1_num} {label[0]} patches")
    print(f"{label_2_num} {label[1]} patches")
