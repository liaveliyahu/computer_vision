# CIFAR10_with_CNN
<p align="center">
  <img src="Assets\openaigym_cartpole_dqn_agent.gif" width="450">
</p>

## The Data
The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research.[1][2] The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.[3] The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class

## The Model
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 15, 15, 32)        896
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 64)          18496
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0
_________________________________________________________________
dropout (Dropout)            (None, 1152)              0
_________________________________________________________________
dense (Dense)                (None, 1024)              1180672
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                10250
=================================================================
Total params: 1,284,170
Trainable params: 1,284,170
Non-trainable params: 0
_________________________________________________________________

## Results
### Loss
<p align="center">
  <img src="Assets\loss.png" width="450">
</p>

### Accuracy
<p align="center">
  <img src="Assets\accuracy.png" width="450">
</p>

### Confusion Matrix
<p align="center">
  <img src="Assets\confusion_matrix.png" width="450">
</p>

## Predictions
<p align="center">
  <img src="Assets\predictions.png" width="450">
</p>
