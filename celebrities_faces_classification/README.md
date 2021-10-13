# celebrities_faces_classification
<p align="left">
  <img src="Assets\celebrity.png" width="450">
</p>

## The Data
The data includes images of 14 celebreties faces I got from Kaggle.
There are around 17,200 images in total that I splited into validation and train set.
Each image has 100x100x3 pixels (color images).
I uploaded here only a sample of 7 images per celebrity but anyone can download the full data from: https://www.kaggle.com/danupnelson/14-celebrity-faces-dataset.

## The Model

| Layer (type)                 | Output Shape         | Param #  |
| ---------------------------- | -------------------- | -------- |
| conv2d (Conv2D)              | (None, 100, 100, 32) | 896      |
| batch_normalization (BatchNo | (None, 100, 100, 32) | 128      |
| conv2d_1 (Conv2D)            | (None, 100, 100, 32) | 9248     |
| batch_normalization_1 (Batch | (None, 100, 100, 32) | 128      |
| max_pooling2d (MaxPooling2D) | (None, 50, 50, 32)   | 0        |
| conv2d_2 (Conv2D)            | (None, 50, 50, 64)   | 18496    |
| batch_normalization_2 (Batch | (None, 50, 50, 64)   | 256      |
| conv2d_3 (Conv2D)            | (None, 50, 50, 64)   | 36928    |
| batch_normalization_3 (Batch | (None, 50, 50, 64)   | 256      |
| max_pooling2d_1 (MaxPooling2 | (None, 25, 25, 64)   | 0        |
| conv2d_4 (Conv2D)            | (None, 25, 25, 128)  | 73856    |
| batch_normalization_4 (Batch | (None, 25, 25, 128)  | 512      |
| conv2d_5 (Conv2D)            | (None, 25, 25, 128)  | 147584   |
| batch_normalization_5 (Batch | (None, 25, 25, 128)  | 512      |
| max_pooling2d_2 (MaxPooling2 | (None, 12, 12, 128)  | 0        |
| flatten (Flatten)            | (None, 18432)        | 0        |
| dropout (Dropout)            | (None, 18432)        | 0        |
| dense (Dense)                | (None, 1024)         | 18875392 |
| dropout_1 (Dropout)          | (None, 1024)         | 0        |
| dense_1 (Dense)              | (None, 14)           | 14350    |

Total params: 19,178,542
Trainable params: 19,177,646
Non-trainable params: 896

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
