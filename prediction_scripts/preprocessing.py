"""
Module for extracting needed info (image path, location, timestamp) from each GeoPose,
and performing image transformations.
"""

import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

def extract_info_from_geopose(geopose_item):
    """
    Extract basic info (image_path, longitude, latitude, valid_time) from a single GeoPose item.
    """
    outer_params = geopose_item["outerFrame"]["parameters"]
    longitude = outer_params["longitude"]
    latitude = outer_params["latitude"]

    stream_elem = geopose_item["streamElements"][0]
    valid_time = stream_elem["validTime"]
    image_url = stream_elem["media"]["image_url"]

    return image_url, longitude, latitude, valid_time


def load_and_preprocess_image(image_path, input_size=224, use_crop=False):
    """
    Load and preprocess an image for the model.
    If use_crop is True, you may apply a custom transform that crops the bottom 1/3, etc.
    This example only resizes and normalizes.
    """
    transform_list = []
    # Optionally, if you want to crop bottom 1/3:
    if use_crop:
        transform_list.append(CropBottomOneThird())

    transform_list.append(T.Resize((input_size, input_size)))
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))

    transform = T.Compose(transform_list)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(pil_img)
    return img_tensor


class CropBottomOneThird:
    """
    Custom transform to crop the bottom 1/3 of the image.
    """
    def __call__(self, img):
        width, height = img.size
        crop_y = height // 3
        return img.crop((0, 0, width, height - crop_y))
