# Assignment 7: Transfer Learning

## Objective
Implement Transfer Learning using pre-trained Alexnet, VGG-16, RESNET50, EfficientNetB0 models for image classification and compare their performance.

## Description
This assignment demonstrates the power of Transfer Learning on the CIFAR-10 image classification dataset. It utilizes pre-trained models from `tensorflow.keras.applications`, specifically:
* VGG-16
* ResNet50
* EfficientNetB0

The base models are frozen, and a custom dense classification head is added and trained. The performance of each architecture is then compared visually using a bar chart.

## Files Included
* `Assignment_7_Transfer_Learning.ipynb`: The Jupyter Notebook with the transfer learning pipeline.
* `Assignment_7_Document.docx`: The detailed laboratory report including code and output screenshots.
