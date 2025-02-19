# HyperFlow: Next-Generation Computational Framework for Machine Learning & Deep Learning

## 📦 Official Package

HyperFlow is officially available on PyPI: [HyperFlow-Package](https://pypi.org/project/HyperFlow-Package/)

![Project Image](https://github.com/Shyanil/HyperFlow/blob/main/The%20HyperFlow/HyperFlow_FlowUnit.png)

📄 **Full Documentation:** [Read Here](https://drive.google.com/file/d/1DfNKxcDb-VEAI13xm1UwnLyZwF5_1_t_/view?usp=sharing)

## 📌 Introduction

HyperFlow is an advanced computational framework designed to streamline machine learning and deep learning research. At its core, the **FlowUnit** class provides an optimized, modular implementation for:

- **Mathematical operations**
- **Activation functions**
- **Optimization techniques**
- **Backpropagation**

HyperFlow aims to combine simplicity with power, making it an ideal tool for understanding neural networks, automatic differentiation, and optimization techniques.

## 🎯 Inspiration

HyperFlow is inspired by [Micrograd](https://github.com/karpathy/micrograd) by **Andrej Karpathy** but extends its functionality by offering:

- Advanced mathematical operations
- A wider range of activation functions
- Optimized backpropagation
- Enhanced modularity

It serves as a **lightweight yet powerful alternative** to more complex deep learning frameworks like PyTorch.

## 🚀 Why Choose HyperFlow?

✅ **Lightweight & Transparent** – Uses raw Python implementations for better ML/DL understanding.  
⚡ **Efficient & Optimized** – Uses mapping functions for performance improvements.  
🔧 **Flexible & Powerful** – Supports neural networks, including backpropagation.  
📉 **Minimal Dependencies** – Encourages learning without over-reliance on pre-built libraries.

## 🔍 Core Functionalities

### 🔢 Mathematical Operations

- `create2darray`, `convert2darray`, `add`, `sub`, `mul`, `matmul`, `dot`, `pow`

### ⚙️ Activation Functions

- `sigmoid`, `tanh`, `ReLU`, `Leaky ReLU`, `softmax`

### 📉 Loss Functions

- `categorical_cross_entropy`, `binary_cross_entropy`, `mse_loss`

### 🔄 Optimization & Backpropagation

- `backpropagate`, `gradient_descent`

## 🧠 Neural Network Implementation

The **Neuron.py** module simplifies neural network development, allowing users to create:

- **Neurons**
- **Layers**
- **Complete Neural Networks**

This module provides full control over weights, biases, and architecture, enabling in-depth experimentation.

## 🛠 Installation & Usage

### 1️⃣ Install HyperFlow

Run the following command in a Jupyter Notebook or terminal:

```sh
pip install HyperFlow-Package
```

### 2️⃣ Import HyperFlow Modules

```python
from HyperFlow.src.FlowUnit_module import FlowUnit, LossFunctions
from HyperFlow.src.Nuron import *
```

### 3️⃣ Example Usage

```python
import time

# Create FlowUnit instances
a = FlowUnit([1, 2, 3])
b = FlowUnit([4, 5, 6])

# Compute the dot product
result = a.__dot__(b)

# Measure execution time
start_time = time.time()
print(f"Dot product result: {result.data}")
end_time = time.time()
py_time = end_time - start_time
print(f"Time taken: {py_time:.6f} seconds")
```

#### 📌 Expected Output:
```
Dot product result: 32
Time taken: 0.000421 seconds
```

### 4️⃣ Backpropagation Example

```python
def test_backpropagation():
    """Test backpropagation through multiple FlowUnit operations."""
    x = FlowUnit(2.0)
    y = FlowUnit(-3.0)
    z = FlowUnit(1.5)

    # Forward computations
    a = x.sigmoid()
    b = y.tanh()
    c = z.relu()
    d = x.leaky_relu()

    # Dummy loss function: Sum of all outputs
    loss = a + b + c + d

    # Backpropagation
    loss.backpropagate()

    # Print gradients
    print(f"x.grad: {x.grad}")
    print(f"y.grad: {y.grad}")
    print(f"z.grad: {z.grad}")
    print(f"a.grad (sigmoid): {a.grad}")
    print(f"b.grad (tanh): {b.grad}")
    print(f"c.grad (ReLU): {c.grad}")
    print(f"d.grad (Leaky ReLU): {d.grad}")

# Run the test function
test_backpropagation()
```

#### 📌 Expected Output:
```
x.grad: 1.1049935854035067
y.grad: 0.00986603716543999
z.grad: 1.0
a.grad (sigmoid): 1.0
b.grad (tanh): 1.0
c.grad (ReLU): 1.0
d.grad (Leaky ReLU): 1.0
```

For additional test cases, refer to **test.ipynb**.

## 📚 Additional Resources

- For a **detailed explanation**, refer to the full documentation:  
  [Read Documentation](https://drive.google.com/file/d/1DfNKxcDb-VEAI13xm1UwnLyZwF5_1_t_/view)

## 🤝 Contribution & License

🔹 **Contributions are welcome!** Please read the documentation before making changes.  
🔹 **License:** MIT License – Feel free to use and contribute! 🚀

---

### 🔗 Connect with the Community

Have questions or suggestions? Feel free to open an issue or contribute to the repository!

🚀 **Let's build better ML models together with HyperFlow!**
