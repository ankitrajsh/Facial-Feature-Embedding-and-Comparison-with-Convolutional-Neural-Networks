# Face Image Embedding and Matching

## Overview

This project focuses on processing face images to generate embeddings and compare them using a Convolutional Neural Network (CNN) model. The process involves normalizing image brightness and contrast, extracting embeddings using the DeepFace library, and comparing these embeddings to calculate the matching percentage between two images.

## Project Components

### 1. Image Processing

#### `normalize_brightness_contrast_grayscale(image)`

This function processes the input image to normalize its brightness, contrast, and convert it to grayscale. The steps involved are:

- **Brightness Normalization**: Converts the image from BGR to LAB color space, applies histogram equalization on the L channel, and converts it back to BGR.
- **Contrast Normalization**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel to enhance contrast.
- **Grayscale Conversion**: Converts the image to grayscale after normalization.

### 2. Embedding Extraction

#### `get_all_embeddings(img_path)`

This function extracts face embeddings from an image using the DeepFace library. The steps are:

- **Image Reading**: Reads the image from the specified path.
- **Processing**: Normalizes brightness, contrast, and converts to grayscale.
- **Embedding Extraction**: Uses DeepFace with the `Facenet512` model to generate an embedding for the processed image.
- **Temporary File Handling**: Saves the processed image as a temporary file, extracts the embedding, and deletes the temporary file.

### 3. CNN Model

#### `CNNModel` Class

Defines a simple 1D convolutional neural network for processing embeddings.

- **Architecture**:
  - **Layer**: A single 1D convolutional layer (`nn.Conv1d`) that reduces the dimensionality of the embedding from 512 to 128.

#### `get_CNN_embeddings(model, embed)`

This function processes embeddings through the CNN model:

- **Tensor Conversion**: Converts the embedding to a PyTorch tensor.
- **Batch Dimension**: Adds a batch dimension if necessary.
- **Model Processing**: Passes the tensor through the CNN model to obtain processed embeddings.
- **Output Handling**: Converts the output tensor back to a NumPy array or list.

### 4. Matching Percentage Calculation

#### `matching_percentage(img_path1, img_path2)`

Compares embeddings of two images and calculates the percentage of matching elements:

- **Embedding Extraction**: Extracts embeddings for both images.
- **CNN Processing**: Processes the embeddings through the CNN model.
- **Matching Calculation**: Calculates the number of matching elements and computes the percentage of matching elements between the two processed embeddings.

### 5. Directory Handling

- **Image Directory Setup**: Lists all images in a specified directory and its subfolders to gather images for processing and embedding extraction.

## Usage

1. **Install Dependencies**

   Ensure you have the required libraries installed. Use the following command to install them:

   ```bash
   pip install opencv-python deepface torch numpy
