# Image Segmentation and Generation Guide

## Introduction
This guide provides instructions on using the segmentation and generation tools for processing and augmenting image datasets. The code leverages the `Segmentor` and `Generator` classes to perform image segmentation and overlay generation respectively. Follow the steps below to get started, understand the allowed inputs, and utilize each segmentor and generator effectively.

## Getting Started

1. **Install Required Dependencies:**
   Ensure you have all necessary dependencies installed. Typically, this would involve packages like `opencv`, `numpy`, and any other specific libraries used by `Segmentor` and `Generator`.

   ```bash
   pip install -r requirements.txt
   ```

## Segmentor

### Introduction
The `Segmentor` class is designed to segment images based on provided masks. Segmentation involves identifying and isolating specific parts of an image.

### Allowed Inputs
- `source_images`: Path to the directory, list of image paths or a single path containing images to be segmented.
- `source_masks`: Path to the directory, list of image paths or a single path containing segmentation masks.
- `destination_directory`: Path to the *directory* where segmented images will be saved.

### Usage

1. **Import and Initialize Segmentor:**

    ```python
    from code.segmentor import Segmentor

    segmentor = Segmentor()
    ```

2. **Segment Images:**

    ```python
    segmentor.segment(
        source_images='./DS/images/',
        source_masks='./DS/segments/',
        destination_directory='./output/segmented'
    )
    ```

3. **Check Output:**
   The segmented images will be saved in the `./output/segmented` directory.

## Generator

### Introduction
The `Generator` class is designed to create new images by overlaying segmented images onto different backgrounds. This is useful for data augmentation and generating varied datasets.

### Allowed Inputs
- `backgrounds`: Path to the directory containing background images.
- `overlays`: Path to the directory containing overlay images (segmented images).
- `destination_directory`: Path to the directory where generated overlay images will be saved.

### Usage

1. **Import and Initialize Generator:**

    ```python
    from code.generator import Generator

    generator = Generator()
    ```

2. **Generate Overlay Images:**

    ```python
    generator.generate(
        backgrounds='./DS/backgrounds',
        overlays='./output/segmented',
        destination_directory='./output/overlay_backgrounds'
    )
    ```

3. **Check Output:**
   The generated overlay images will be saved in the `./output/overlay_backgrounds` directory in three category, `results`, `results_mask` and `yolo_label`

## Conclusion
This guide provides the basic steps to use the `Segmentor` and `Generator` classes for segmenting images and generating augmented datasets. Ensure that your directories are correctly structured and paths are correctly specified to avoid errors. For advanced usage and customization, refer to the respective class documentation and source code.