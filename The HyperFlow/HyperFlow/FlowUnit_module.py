import math
import numpy as np
from itertools import chain


class FlowUnit:
    def __init__(self, data, childs=(), op="", label=""):
        self.data = data
        self._prev = set([child for child in childs if isinstance(child, FlowUnit)])
        self.op = op
        self.label = label
        self.grad = 0.0
        self.backward = lambda: None

    def create2d_array(start, end, rows, cols):
        """
        Creates a 2D array with values linearly spaced between `start` and `end`.
        Args:
        start (float): The starting value of the array.
        end (float): The ending value of the array.
        rows (int): The number of rows in the 2D array.
        cols (int): The number of columns in the 2D array.

        Returns:
        FlowUnit: A FlowUnit object containing the generated 2D numpy array.
        """
        total_elements = rows * cols
        step = (end - start) / (total_elements - 1)
        data = []
        current = start
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(current)
                current += step
            data.append(row)
        return FlowUnit(np.array(data))

    def convert2d_array(values, rows, cols):
        """
        Converts a 1D array of values into a 2D array with specified rows and columns.
        Args:
        values (list): The 1D array of values to be reshaped.
        rows (int): The number of rows in the resulting 2D array.
        cols (int): The number of columns in the resulting 2D array.

        Returns:
        FlowUnit: A FlowUnit object containing the reshaped 2D numpy array.

        Raises:
        ValueError: If the total number of values does not match `rows * cols`.
        """
        if len(values) != rows * cols:
            raise ValueError(f"Total values must equal {rows * cols}")
        data = []
        for i in range(rows):
            row = values[i * cols : (i + 1) * cols]
            data.append(row)
        return FlowUnit(np.array(data))

    def __repr__(self):
        """
        Returns a string representation of the FlowUnit object.

        If the `data` attribute is a numpy array:
            - If it is a 2D array or has more than one element, the string representation includes the full array.
            - If it is a 1D array with a single element, only that element is shown.

        Args:
        None

        Returns:
        str: The string representation of the FlowUnit object.
        """
        if isinstance(self.data, np.ndarray):
            if self.data.ndim == 2 or len(self.data) > 1:
                return f"FlowUnit(\n{np.array2string(self.data)})"
            else:
                return f"FlowUnit({self.data[0]})"
        else:
            return f"FlowUnit({self.data})"

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        """
        Defines the addition operator for FlowUnit objects.

        Adds the `data` attribute of two FlowUnit objects or a FlowUnit object and another value.

        Args:
        other (FlowUnit or numeric): The other operand to be added to the current FlowUnit object.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the addition.

        Note:
        - Defines a backward function that updates the `grad` attribute of both FlowUnit objects.
        """
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data + other.data, (self, other), label="+")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out.backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """
        Defines the multiplication operator for FlowUnit objects.

        Multiplies the `data` attribute of two FlowUnit objects or a FlowUnit object and another value.

        Args:
        other (FlowUnit or numeric): The other operand to be multiplied with the current FlowUnit object.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the multiplication.

        Note:
        - A `backward` function is defined to update the `grad` attribute of both FlowUnit objects.
        """

        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data * other.data, (self, other), label="*")

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = backward
        return out

    def _matmul(self, other_data):
        """
        Defines matrix multiplication for FlowUnit objects.

        The `_matmul` method performs the core matrix multiplication for two numpy arrays.

        Args:
        other_data (numpy.ndarray): The second operand's data for matrix multiplication.

        Returns:
        numpy.ndarray: The result of the matrix multiplication.

        Raises:
        ValueError: If the shapes of the matrices are not compatible for multiplication.

        The `__matmul__` method enables the use of the `@` operator for matrix multiplication.

        Args:
        other (FlowUnit or numeric): The operand to multiply with the current FlowUnit object.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the matrix multiplication.

        Note:
        - A `backward` function is defined to compute and propagate gradients manually during backpropagation.
        """

        if self.data.shape[1] != other_data.shape[0]:
            raise ValueError("Shapes must be (a, b) @ (b, c) for matmul.")

        result = [
            [sum(map(lambda a, b: a * b, row, col)) for col in other_data.T]
            for row in self.data
        ]
        return np.array(result)

    def __matmul__(self, other):
        if not isinstance(other, FlowUnit):
            other = FlowUnit(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("Both operands must be 2D arrays.")

        out_data = self._matmul(other.data)
        out = FlowUnit(out_data, (self, other), label="@")

        def backward():
            grad_self = [
                [
                    sum(map(lambda x, y: x * y, row_out_grad, col))
                    for col in other.data.T
                ]
                for row_out_grad in out.grad
            ]
            self.grad += np.array(grad_self)

            grad_other = [
                [
                    sum(map(lambda x, y: x * y, row, col_out_grad))
                    for col_out_grad in out.grad.T
                ]
                for row in self.data
            ]
            other.grad += np.array(grad_other)

        out.backward = backward
        return out

    def _dot_product(self, other_data):
        """
        Defines the dot product for FlowUnit objects.

        The `_dot_product` method calculates the dot product between two 1D numpy arrays (vectors).

        Args:
        other_data (numpy.ndarray): The second operand's data for the dot product.

        Returns:
        float: The result of the dot product.

        Raises:
        ValueError: If the vectors have different lengths.

        The `__dot__` method enables the use of the `.dot` operator for dot product calculation.

        Args:
        other (FlowUnit or numeric): The operand to compute the dot product with the current FlowUnit object.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the dot product.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """
        if len(self.data) != len(other_data):
            raise ValueError("Vectors must have the same length")

        return sum(map(lambda a, b: a * b, self.data, other_data))

    def __dot__(self, other):
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        if (isinstance(self.data, np.ndarray) and self.data.ndim != 1) or (
            isinstance(other.data, np.ndarray) and other.data.ndim != 1
        ):
            raise ValueError("Dot product operands must be vectors (1D arrays).")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must have the same length")
        out_data = self._dot_product(other.data)
        out = FlowUnit(out_data, (self, other), label="dot")

        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out.backward = backward
        return out

    def __sub__(self, other):
        """
        Defines the subtraction operator for FlowUnit objects.

        Subtracts the `data` attribute of two FlowUnit objects or a FlowUnit object and another value.

        Args:
        other (FlowUnit or numeric): The other operand to be subtracted from the current FlowUnit object.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the subtraction.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """

        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(
            self.data - other.data, (self, other), label="-"
        )  # Correct subtraction

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out.backward = backward
        return out

    def __truediv__(self, other):
        """
        Defines the true division operator for FlowUnit objects.

        Divides the `data` attribute of one FlowUnit object by another or a FlowUnit object by a numeric value.

        Args:
        other (FlowUnit or numeric): The divisor operand.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the division.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data / other.data, (self, other), label="/")

        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad

        out.backward = backward
        return out

    def pow(self, other):
        """
        Defines the power operation for FlowUnit objects.

        Raises the `data` attribute of the FlowUnit object to the power of a numeric value.

        Args:
        other (float or int): The exponent to raise the `data` attribute to.

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the power operation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """

        assert isinstance(other, (float, int)), ValueError(
            "Only Float and Integer allowed"
        )
        out = FlowUnit(self.data**other, (self,), label=f"**{other}")

        def backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out.backward = backward
        return out

    def mean(self):
        """
        Calculates the mean of the `data` attribute of the FlowUnit object.

        If the `data` is a list, the mean is calculated as the sum of elements divided by the length.
        If the `data` is a numpy array, `np.mean` is used to compute the mean.

        Args:
        None

        Returns:
        float: The mean of the data.

        Raises:
        TypeError: If the `data` attribute is neither a list nor a numpy array.
        """

        if isinstance(self.data, list):
            return sum(self.data) / len(self.data)
        elif isinstance(self.data, np.ndarray):
            return np.mean(self.data)
        else:
            raise TypeError("Unsupported data type for mean calculation")

    def log(self):
        return FlowUnit(np.log(self.data), (self,), label="log")

    def __len__(self):
        """Override len function to return the length of data if possible"""
        if hasattr(self.data, "__len__"):  # Check if self.data supports len()
            return len(self.data)
        else:
            return 1

    def __rsub__(self, other):
        """Handles something - FlowUnit"""
        return FlowUnit(other - self.data, (self,), label="-")

    def __neg__(self):
        """Unary minus (-FlowUnit)"""
        return FlowUnit(-self.data, (self,), label="-")

    def reshape(self, *shape):
        """Reshape the data attribute using numpy's reshape"""
        return FlowUnit(self.data.reshape(*shape))

    def __lt__(self, other):
        if isinstance(other, FlowUnit):
            return (
                self.data < other.data
            )  # Assuming `FlowUnit` has a `.value` attribute
        return self.data < other

    def sigmoid(self):
        """
        Applies the sigmoid activation function to the `data` attribute of the FlowUnit object.

        The sigmoid function is computed as 1 / (1 + exp(-data)).

        Args:
        None

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the sigmoid activation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """
        out_data = 1 / (1 + np.exp(-self.data))
        out = FlowUnit(out_data, (self,), label="sigmoid")

        def backward():
            self.grad += (out.data * (1 - out.data)) * out.grad

        out.backward = backward
        return out

    def tanh(self):
        """
        Applies the hyperbolic tangent (tanh) activation function to the `data` attribute of the FlowUnit object.

        The tanh function is computed as (exp(2 * data) - 1) / (exp(2 * data) + 1).

        Args:
        None

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the tanh activation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """

        exp_2x = np.exp(2 * self.data)
        out_data = (exp_2x - 1) / (exp_2x + 1)
        out = FlowUnit(out_data, (self,), label="tanh")

        def backward():
            self.grad += (1 - out_data**2) * out.grad

        out.backward = backward
        return out

    def relu(self):
        """
        Applies the ReLU (Rectified Linear Unit) activation function to the `data` attribute of the FlowUnit object.

        The ReLU function is computed as the element-wise maximum of 0 and the `data`.

        Args:
        None

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the ReLU activation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """

        out_data = np.maximum(0, self.data)
        out = FlowUnit(out_data, (self,), label="relu")

        def backward():
            self.grad += np.where(self.data > 0, 1.0, 0.0) * out.grad

        out.backward = backward
        return out

    def leaky_relu(self, alpha=0.01):
        """
        Applies the Leaky ReLU activation function to the `data` attribute of the FlowUnit object.

        The Leaky ReLU function is computed as:
        - `data` if `data > 0`
        - `alpha * data` if `data <= 0`

        Args:
        alpha (float, optional): The slope for negative values (default is 0.01).

        Returns:
        FlowUnit: A new FlowUnit object containing the result of the Leaky ReLU activation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        """

        out_data = np.where(self.data > 0, self.data, alpha * self.data)
        out = FlowUnit(out_data, (self,), label="leaky_relu")

        def backward():
            self.grad += np.where(self.data > 0, 1, alpha) * out.grad

        out.backward = backward
        return out

    def softmax(self):
        """
        Applies the softmax activation function to the `data` attribute of the FlowUnit object.

        The softmax function computes the probabilities of each value in the `data` array (or list) as:
        - exp(data[i] - max(data)) / sum(exp(data[i] - max(data)))

        This ensures numerical stability by subtracting the maximum value before exponentiating.

        Args:
        None

        Returns:
        FlowUnit: A new FlowUnit object containing the probabilities as the result of the softmax activation.

        Note:
        - A `backward` function is defined to compute and propagate gradients during backpropagation.
        - The backward pass computes gradients using the chain rule for the softmax function.
        """

        """Compute softmax on the FlowUnit's data (list of logits)"""
        z = self.data

        if isinstance(z, list) and all(isinstance(i, list) for i in z):
            z = list(chain.from_iterable(z))
        elif isinstance(z, np.ndarray):
            z = z.tolist()
        elif not isinstance(z, list):
            raise ValueError(f"Input data must be list or array-like, got {type(z)}")

        if not z:
            raise ValueError("Input data cannot be empty.")

        max_z = max(z)
        exp_z = [math.exp(zi - max_z) for zi in z]
        sum_exp_z = sum(exp_z)
        probs = [ez / sum_exp_z for ez in exp_z]

        out = FlowUnit(probs, (self,), label="softmax")
        out.grad = [0.0] * len(probs)

        def backward():
            """Backward pass for softmax"""
            n = len(probs)
            if isinstance(out.grad, float):
                out.grad = [out.grad] * n

            self.grad = [0.0] * n

            for i in range(n):
                for j in range(n):
                    if i == j:
                        self.grad[i] += probs[i] * (1 - probs[j]) * out.grad[j]
                    else:
                        self.grad[i] += -probs[i] * probs[j] * out.grad[j]

        out.backward = backward
        return out

    def backpropagate(self):
        """
        Performs backpropagation through the computation graph, starting from the current FlowUnit.

        This method computes the gradients for all the FlowUnit objects in the graph by:
        1. Building a topological order of nodes (from input to output).
        2. Setting the initial gradient for the current node.
        3. Iterating through the nodes in reverse order and calling the `backward()` method on each node to propagate gradients.

        Args:
        None

        Returns:
        None

        Note:
        - The topological order ensures that the backward pass is performed in the correct order of dependencies.
        - The backward pass calculates gradients using the chain rule for each node.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node.backward()

    def gradient_descent(parameters, learning_rate=0.01, iterations=1000):
        """
        Performs gradient descent optimization on the provided parameters.

        This method updates the parameters' `data` attribute by subtracting the gradient multiplied by the learning rate, effectively minimizing the loss function over multiple iterations.

        Args:
        - parameters (list): A list of FlowUnit objects whose gradients will be used for updating the parameters.
        - learning_rate (float, optional): The step size for updating the parameters (default is 0.01).
        - iterations (int, optional): The number of gradient descent iterations (default is 1000).

        Returns:
        None

        Note:
        - The gradients of each parameter must be set before calling this method.
        - The gradients are cleared at the start of each iteration to avoid accumulation.
        """

        for _ in range(iterations):
            parameters[0].grad = 0.0
            for param in parameters:
                param.data -= learning_rate * param.grad


class LossFunctions(FlowUnit):
    @staticmethod
    def categorical_cross_entropy(logits, target, parameters=None, lambda_reg=0.01):
        """
        Compute categorical cross-entropy loss.
        logits: FlowUnit containing raw scores
        target: list representing one-hot encoded target
        """
        if len(logits.data) != len(target):
            raise ValueError("Logits and target must have the same length")

        if sum(target) != 1 or set(target) - {0, 1}:
            raise ValueError("Target must be one-hot encoded")
        probs = logits.softmax()
        epsilon = 1e-8
        loss = -sum(t * math.log(p + epsilon) for t, p in zip(target, probs.data))

        if parameters:
            l2_reg = sum(w.data**2 for w in parameters)
            loss += lambda_reg * l2_reg

        return FlowUnit(loss, (logits,), label="CCE")

    @staticmethod
    def binary_cross_entropy_loss(inputs, target, parameters, lambda_reg=0.01):
        """
        Computes the binary cross-entropy loss with L2 regularization.

        This method calculates the binary cross-entropy loss between predicted and target values, with L2 regularization applied on the parameters (weights and bias).

        Args:
        - inputs (FlowUnit): The input data (features).
        - target (FlowUnit): The true target values.
        - parameters (tuple): A tuple containing the model parameters (weights, bias).
        - lambda_reg (float, optional): The regularization strength (default is 0.01).

        Returns:
        - FlowUnit: The computed loss with regularization.

        Note:
        - Assumes binary classification with sigmoid activation function.
        - Regularization is applied on both weights and bias.
        """
        w, b = parameters
        if inputs.data.ndim == 1:
            inputs.data = inputs.data.reshape(-1, 1)
        if w.data.ndim == 1:
            w.data = w.data.reshape(-1, 1)
        z = inputs.__matmul__(w)
        b_broadcasted = FlowUnit(np.ones_like(z.data) * b.data)
        z += b_broadcasted
        f_wb = z.sigmoid()
        target_flow = FlowUnit(target.data.reshape(-1, 1))

        loss = -(target_flow * f_wb.log() + (1 - target_flow) * (1 - f_wb).log())
        loss = loss.mean()
        l2_reg = lambda_reg * (w.pow(2) + b.pow(2))
        out = (loss + l2_reg).data[0]
        return FlowUnit(out, (inputs, target_flow), label="BCE")

    @staticmethod
    def mse_loss(X, y, parameters, lambda_=1):
        """
        Computes the cost over all examples
        Args:
          X (FlowUnit): Data, m examples with n features
          y (FlowUnit): target values
          parameters (tuple): model parameters (w, b)
          lambda_ (scalar): Controls amount of regularization
        Returns:
          total_cost (FlowUnit): cost wrapped in a FlowUnit instance
        """
        w, b = parameters
        X_data = X.data
        if isinstance(y, FlowUnit):
            y_data = y.data
        else:
            y_data = np.array(y)
        w_data = w.data
        b_data = b.data
        m = X_data.shape[0]
        n = X_data.shape[1] if len(X_data.shape) > 1 else 1
        cost = 0
        for i in range(m):
            f_wb_i = (X_data[i].__matmul__(w_data)) + b_data
            cost += (f_wb_i - y_data[i]) ** 2
        cost = cost / (2 * m)
        reg_cost = sum(w_data**2) * lambda_ / (2 * m)
        total_cost = cost + reg_cost
        return FlowUnit(total_cost, (X, y), label="MSE")
