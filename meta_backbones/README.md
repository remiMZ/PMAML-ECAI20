# Backbones

Several backbone networks are provided:

* `con1.py`: Conv-1
* `con2.py`: Conv-2
* `con4.py`: Conv-4
* `con6.py`: Conv-6
* `con8.py`: Conv-8
* `resnet18.py`: ResNet-18

## Introduction

### Conv-1

This architecture is composed of 1 convolutional blocks. Each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.

### Conv-2

This architecture is composed of 2 convolutional blocks. Each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.

### Conv-4

This architecture is composed of 4 convolutional blocks. Each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.

Used in "Matching Networks for One Shot Learning, NIPS 2016" and "Prototypical Networks for Few-shot Learning, NIPS 2017".

### Conv-6

This architecture is composed of 6 convolutional blocks. In the first four blocks, each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.
While the last two blocks, each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity.

### Conv-8

This architecture is composed of 8 convolutional blocks. In the first four blocks, each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity and a 2x2 max-pooling layer.
While the last four blocks, each block comprises a 64-filter 3x3 convolution, a batch normalization layer, a ReLU nonlinearity.

### ResNet-18

包含 resnet10, resnet18

