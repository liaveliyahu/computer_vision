# CIFAR10_with_CNN
<p align="left">
  <img src="Assets\cifar10.jpg" width="450">
</p>

## The Data
The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

## The Model

| Layer (type)        | Output Shape       |  Param # |
| ------------------- | ------------------ | -------- |
| conv2d (Conv2D)     | (None, 15, 15, 32) | 896      |
| conv2d_1 (Conv2D)   | (None, 7, 7, 64)   | 18496    |
| conv2d_2 (Conv2D)   | (None, 3, 3, 128)  | 73856    |
| flatten (Flatten)   | (None, 1152)       | 0        |
| dropout (Dropout)   | (None, 1152)       | 0        |
| dense (Dense)       | (None, 1024)       | 1180672  |
| dropout_1 (Dropout) | (None, 1024)       | 0        |
| dense_1 (Dense)     | (None, 10)         | 10250    |

Total params: 1,284,170                             
Trainable params: 1,284,170                         
Non-trainable params: 0                             

## Results
### Loss
<p align="left">
  <img src="Assets\loss.png" width="450">
</p>

### Accuracy
<p align="left">
  <img src="Assets\accuracy.png" width="450">
</p>

### Confusion Matrix
<p align="left">
  <img src="Assets\confusion_matrix.png" width="450">
</p>

## Predictions
<p align="left">
  <img src="Assets\predictions.png" width="450">
</p>
