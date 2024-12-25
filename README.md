# Calculation of Object Dimensions

This project utilizes the YOLOv8 model to detect a TV and calculate its dimensions in real-time or from a set of images. It outputs the width and height of the detected TV in pixels, providing valuable information for various applications.



## Table of Contents

 - [Introduction](#Introduction)
 - [Features](#Features)
 - [How to run](#Howtorun)
 


## Introduction

This project aims to demonstrate how to use the YOLOv8 object detection model to identify and measure the dimensions of a TV. By leveraging a pre-trained YOLOv8 model, the program can detect TVs in real-time using a webcam or process images from a directory.


## Features

- Real-time TV detection using a webcam.
- Calculation of TV dimensions in pixels.
- Easy-to-follow scripts for training and validating the model.
- Predicting dimensions on test images.


## How to run 

1. Install dependencies:

```bash
  Pip install ultralytics
```
2. Start Image prediction:

```bash
  Run img.py for img detection
```
3. Camera prediction:

```bash
  Run a.py for camera detection
```
## Note

- training.py file is used to custom training for the dataset.
- validate.py file is used to validate custom model.
- predict.py file is used to inference with custom model.