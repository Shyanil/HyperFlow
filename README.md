# HyperFlow: Next-Generation Computational Framework for Machine Learning & Deep Learning

![Project Image](https://github.com/Shyanil/HyperFlow/blob/main/The%20HyperFlow/HyperFlow_FlowUnit.png)

📄 [Read the Full Documentation](PDF_LINK_HERE)

## Introduction

HyperFlow is an advanced computational framework designed to enhance machine learning and deep learning research. At its core is the **FlowUnit** class, an optimized and modular implementation supporting:

- Mathematical operations
- Activation functions
- Optimization techniques
- Backpropagation

HyperFlow provides simplicity and power, making it ideal for understanding neural networks, automatic differentiation, and optimization techniques.

## Inspiration

Inspired by [Micrograd](https://github.com/karpathy/micrograd) by **Andrej Karpathy**, HyperFlow extends its capabilities by offering:

- Advanced mathematical operations
- Broader activation function support
- Optimized backpropagation
- Enhanced modularity

It serves as a lightweight yet powerful alternative to complex deep learning frameworks like PyTorch.

## Why HyperFlow?

✅ **Lightweight & Transparent** – Focuses on raw Python implementations to help understand ML/DL concepts.  
⚡ **Efficient & Optimized** – Uses map functions for better performance.  
🔧 **Flexible & Powerful** – Supports neural networks, including backpropagation.  
📉 **Minimal NumPy Dependency** – Encourages learning without excessive reliance on pre-built libraries.

## Core Functionalities

### 🔢 Mathematical Operations

- `create2darray`, `convert2darray`, `add`, `sub`, `mul`, `matmul`, `dot`, `pow`

### ⚙️ Activation Functions

- `sigmoid`, `tanh`, `ReLU`, `Leaky ReLU`, `softmax`

### 📉 Loss Functions

- `categorical_cross_entropy`, `binary_cross_entropy`, `mse_loss`

### 🔄 Optimization & Backpropagation

- `backpropagate`, `gradient_descent`

## 🧠 Neural Network Implementation

The **Neuron.py** module simplifies the creation of:

- Neurons
- Layers
- Complete Neural Networks

This module offers full control over weights, biases, and architecture for in-depth experimentation.

## Contribution & License

To contribute, **read the documentation carefully** before working with the code.  
**License:** MIT License – Feel free to use and contribute! 🚀
