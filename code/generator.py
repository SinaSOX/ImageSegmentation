import cv2 as cv
import numpy as np
import random
import os
from typing import Union
from tqdm import tqdm

class Generator:
    def __init__(self) -> None:
        self.overlay_images = None
        self.background_images = None

    def generate(self, backgrounds: Union[list[str], str], overlays: Union[list[str], str], destination_directory: str, max_overlay=10):
        """
        Generate images with overlay on backgrounds and save them to the destination directory.

        Args:
            backgrounds (Union[list[str], str]): List of background image paths or a directory containing background images.
            overlays (Union[list[str], str]): List of overlay image paths or a directory containing overlay images.
            destination_directory (str): Directory where the generated images and labels will be saved.
            max_overlay (int, optional): Maximum number of overlays per background image. Defaults to 10.
        """
        # Load background images
        if isinstance(backgrounds, str):
            if os.path.isdir(backgrounds):
                if not os.path.exists(backgrounds):
                    raise ValueError("Background input folder directory doesn't exist.")
                self.background_images = list(map(lambda img: os.path.join(backgrounds, img), os.listdir(backgrounds)))
            else:
                self.background_images = [backgrounds]
        else:
            self.background_images = backgrounds

        # Load overlay images
        if isinstance(overlays, str):
            if os.path.isdir(overlays):
                if not os.path.exists(overlays):
                    raise ValueError("Overlay input folder directory doesn't exist.")
                self.overlay_images = list(map(lambda img: os.path.join(overlays, img), os.listdir(overlays)))
            else:
                self.overlay_images = [overlays]
        else:
            self.overlay_images = overlays

        # Create destination directories if they don't exist
        if not os.path.exists(destination_directory):
            os.mkdir(destination_directory)
        if not os.path.exists(os.path.join(destination_directory, 'results')):
            os.mkdir(os.path.join(destination_directory, 'results'))
        if not os.path.exists(os.path.join(destination_directory, 'yolo_label')):
            os.mkdir(os.path.join(destination_directory, 'yolo_label'))
        if not os.path.exists(os.path.join(destination_directory, 'results_mask')):
            os.mkdir(os.path.join(destination_directory, 'results_mask'))

        # Process each background image
        for background_path in tqdm(self.background_images, desc="Generating overlay backgrounds:"):
            background_image = cv.imread(background_path)
            overlays = self._load_images(self.overlay_images)

            num_overlays = random.randint(1, max_overlay)
            modified_background, mask, labels = self._overlay_images(background_image, overlays, num_overlays)

            output_image_path = os.path.join(os.path.join(destination_directory, 'results'), os.path.basename(background_path).split('.')[0] + '.png')
            output_label_path = os.path.join(os.path.join(destination_directory, 'yolo_label'), os.path.basename(background_path).split('.')[0] + '.txt')
            output_mask_path = os.path.join(os.path.join(destination_directory, 'results_mask'), os.path.basename(background_path).split('.')[0] + '.png')
            self._save_image_and_labels(output_image_path, output_label_path, output_mask_path, modified_background, mask, labels)

    def _load_images(self, image_paths):
        """
        Load images from given paths.

        Args:
            image_paths (list[str]): List of image file paths.

        Returns:
            list: List of loaded images.
        """
        images = []
        for path in image_paths:
            img = cv.imread(path, cv.IMREAD_UNCHANGED)
            if img is not None:
                images.append(img)
        return images

    def _overlay_images(self, background, overlays, num_overlays):
        """
        Overlay random images on the background image.

        Args:
            background (ndarray): Background image.
            overlays (list): List of overlay images.
            num_overlays (int): Number of overlays to apply.

        Returns:
            tuple: Modified background image, mask, and labels.
        """
        h_bg, w_bg = background.shape[:2]
        labels = []
        mask = np.zeros((h_bg, w_bg), dtype=np.uint8)

        for i in range(num_overlays):
            overlay = random.choice(overlays)
            h_ov, w_ov = overlay.shape[:2]

            scale_factor = random.uniform(0.1, 0.3)
            new_w = int(w_ov * scale_factor)
            new_h = int(h_ov * scale_factor)

            resized_overlay = cv.resize(overlay, (new_w, new_h), interpolation=cv.INTER_AREA)

            x_offset = random.randint(0, w_bg - new_w)
            y_offset = random.randint(0, h_bg - new_h)

            for c in range(3): 
                background[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                    resized_overlay[:, :, c] * (resized_overlay[:, :, 3] / 255.0) + \
                    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] * (1.0 - resized_overlay[:, :, 3] / 255.0)

            # Update the mask
            alpha_channel = resized_overlay[:, :, 3] / 255.0
            mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = \
                np.maximum(mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w], alpha_channel * 255)

            # Calculate YOLO format labels
            x_center = (x_offset + new_w / 2) / w_bg
            y_center = (y_offset + new_h / 2) / h_bg
            width = new_w / w_bg
            height = new_h / h_bg

            labels.append(f"0 {x_center} {y_center} {width} {height}")

        return background, mask, labels

    def _save_image_and_labels(self, output_image_path, output_label_path, output_mask_path, background, mask, labels):
        """
        Save the generated image, mask, and labels.

        Args:
            output_image_path (str): Path to save the output image.
            output_label_path (str): Path to save the YOLO format labels.
            output_mask_path (str): Path to save the mask image.
            background (ndarray): Modified background image.
            mask (ndarray): Mask image.
            labels (list): List of YOLO format labels.
        """
        cv.imwrite(output_image_path, background)
        cv.imwrite(output_mask_path, mask)
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(labels))
