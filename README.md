# SSD Object Detection Model - YOLOv8 to SSD Conversion

This project includes code for converting YOLOv8 annotations to SSD format, training an SSD (Single Shot MultiBox Detector) model using TensorFlow, and evaluating its performance. The provided code is implemented in a Jupyter Notebook.

## Project Overview

1. **Setting Up and Data Handling:**
    - Defines constants like the maximum number of objects per image (`MAX_OBJECTS_PER_IMAGE`), number of classes (`NUM_CLASSES`), and image dimensions (`img_width`, `img_height`).
    - Functions to:
        - Read YOLOv8 annotations and convert them to SSD format (`read_yolo_annotation`).
        - Create XML annotation files for images (`create_xml_annotation`).
        - Convert an entire dataset from YOLOv8 to SSD format (`convert_dataset`).
    - Functions to:
        - Load and preprocess images and annotations (`load_data`).
        - Preprocess annotations to a format suitable for training (`preprocess_annotations`).

2. **Building the SSD Model:**
    - Defines a function `ssd_model` to build the SSD architecture.
    - The model includes convolutional layers, pooling layers, a fully connected layer, and reshaping layers.
    - The model outputs:
        - Class predictions (using `softmax`).
        - Bounding box predictions (using `sigmoid`).

3. **Training and Evaluation:**
    - Defines the model input shape based on image dimensions.
    - Creates and compiles the SSD model with the Adam optimizer and appropriate loss functions.
    - Trains the model on the preprocessed training dataset.
    - Evaluates the model on the validation dataset and prints performance metrics.

## Cloning the Repository

To clone this repository, run:

```bash
git clone https://github.com/MMZeeshanAI/SSD-Object-Detection-Model.git
```

## Prerequisites

Ensure you have the following libraries installed:

- TensorFlow
- OpenCV
- NumPy
- XML library (Python standard library)

You can install the required libraries using pip:

```bash
pip install tensorflow opencv-python-headless numpy
```

## Dataset

The dataset should be organized as follows:

```
Annotated_clean/
    Annotated/
        labels/
            train/
                *.txt (YOLOv8 format annotations)
            val/
                *.txt (YOLOv8 format annotations)
        images/
            train/
                *.jpg (Training images)
            val/
                *.jpg (Validation images)
        annotations/
            train/ (Converted XML annotations for training images)
            val/ (Converted XML annotations for validation images)
```

## Usage

1. **Data Conversion:**
   Run the notebook to convert YOLOv8 format annotations to SSD format XML annotations.

2. **Load and Preprocess Data:**
   Load and preprocess images and annotations using the provided functions.

3. **Define and Train SSD Model:**
   Define the SSD model architecture and train it on the preprocessed dataset.

4. **Evaluate Model:**
   Evaluate the trained model on the validation dataset.

## Running the Code

Open the `code.ipynb` file in Jupyter Notebook or JupyterLab and execute the cells sequentially.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
