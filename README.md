# Chest X-Ray Pneumonia Detection

This project involves using deep learning to classify chest X-ray images into two categories: Normal and Pneumonia. The dataset used for training and testing the model is sourced from the Kaggle Chest X-Ray Pneumonia dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Project Overview
The project uses the Chest X-ray database from Kaggle, which is divided into two main parts: train and test. Each part contains two folders: NORMAL and PNEUMONIA. The main objective of the project is to build a model to classify the images into these two categories using a Convolutional Neural Network (CNN).

## Dataset
The dataset used for this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle. It contains 5,863 X-ray images categorized into Normal and Pneumonia.

- **Normal**: 1,583 images
- **Pneumonia**: 4,273 images

Dataset source: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

## Model Architecture
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture consists of several convolutional layers followed by max-pooling layers, and fully connected dense layers. The final layer uses softmax activation to output probabilities for the two classes.

## Installation
To run this project locally, you need to have Python installed along with the required libraries. You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Requirements
*TensorFlow
*Numpy
*Pandas
*os
## Usage
Prepare the Dataset:
Download the dataset from Kaggle and place it in a directory named chest_xray.

Run the Training Script:
Execute the Jupyter notebook final.ipynb to preprocess the data, train the model, and evaluate its performance.

Evaluate the Model:
The trained model can be evaluated on the test set to obtain metrics such as accuracy, precision, recall, and F1-score.

Results
The model achieves an accuracy of almost 80% on the test set. Below is a visual representation of precision, recall, and F1-score:


## Contributors
Omar Ashraf Elnabawi, AI Engineer[GitHub Profile](https://github.com/omarelnabawi)
## License
This project is licensed under the MIT License. See the LICENSE file for more details.


### Additional Notes
- **Results**: Replace `path_to_your_image.png` with the actual path to the image file showing precision, recall, and F1-score.
- **Contributors**: Update the GitHub profile link if applicable.
- **License**: Ensure that a LICENSE file is added to the repository if it's not already there.

Feel free to adjust any details or add more specific information as needed. This README template should give a comprehensive overview of your project.
