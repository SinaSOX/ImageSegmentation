import cv2 as cv
import os
from typing import Union
from tqdm import tqdm

class Segmentor:
    def __init__(self) -> None:
        self.images = None
        self.masks = None
        self.image_mask_paired = []
    
    def segment(self, source_images: Union[list[str], str], source_masks: Union[list[str], str], destination_directory: str):
        """
        Segment images using masks and save the results to the destination directory.

        Args:
            source_images (Union[list[str], str]): List of image paths or a directory containing images.
            source_masks (Union[list[str], str]): List of mask paths or a directory containing masks.
            destination_directory (str): Directory where the segmented images will be saved.
        """
        # Load images from directory or list
        if isinstance(source_images, str):
            if os.path.isdir(source_images):
                self.images = list(map(lambda img: os.path.join(source_images, img), os.listdir(source_images)))
            else:
                self.images = [source_images]
        else:
            self.images = source_images
        
        # Load masks from directory or list
        if isinstance(source_masks, str):
            if os.path.isdir(source_masks):
                self.masks = list(map(lambda img: os.path.join(source_masks, img), os.listdir(source_masks)))
            else:
                self.masks = [source_masks]
        else:
            self.masks = source_masks
        
        # Pair images and masks based on their filenames
        for image in self.images:
            for mask in self.masks:
                if os.path.basename(image).split('.')[0] == os.path.basename(mask).split('.')[0]:
                    self.image_mask_paired.append((image, mask))
                    
        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_directory):
            os.mkdir(destination_directory)
                
        # Process each image-mask pair
        for image, mask in tqdm(self.image_mask_paired, desc="Segmentation process"):
            # Load the image
            raw_image = cv.imread(image)
            raw_image = cv.resize(raw_image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)

            # Load the mask and resize it to match the image size
            raw_mask = cv.imread(mask, cv.IMREAD_GRAYSCALE)
            raw_mask = cv.resize(raw_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv.INTER_CUBIC)

            # Apply the mask to the image
            masked_image = cv.bitwise_and(raw_image, raw_image, mask=raw_mask)

            # Convert the masked image to BGRA (add alpha channel)
            bgra_image = cv.cvtColor(masked_image, cv.COLOR_BGR2BGRA)

            # Set the alpha channel based on the mask
            bgra_image[:, :, 3] = raw_mask

            # Save the resulting image as a PNG file
            output_path = os.path.join(destination_directory, os.path.basename(image).split('.')[0] + '.png')
            cv.imwrite(output_path, bgra_image)
