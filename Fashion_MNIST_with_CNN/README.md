# CIFAR10_with_CNN
<p align="left">
  <img src="Assets\fashion_mnist.jpg" width="450">
</p>

## The Data
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## The Model

| Layer (type)                   | Output Shape       | Param # |
| ------------------------------ | ------------------ | ------- |
| conv2d (Conv2D)                | (None, 24, 24, 32) | 832     |
| max_pooling2d (MaxPooling2D)   | (None, 12, 12, 32) | 0       |
| conv2d_1 (Conv2D)              | (None, 8, 8, 64)   | 51264   |
| max_pooling2d_1 (MaxPooling2D) | (None, 4, 4, 64)   | 0       |
| flatten (Flatten)              | (None, 1024)       | 0       |
| dense (Dense)                  | (None, 1024)       | 1049600 |
| dense_1 (Dense)                | (None, 10)         | 10250   |

Total params: 1,111,946
Trainable params: 1,111,946
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
