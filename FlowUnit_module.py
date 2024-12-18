import math
class FlowUnit:
    def __init__(self , data , childs = () , op = "" , label = ""):
        self.data = data
        self._prev = set([child for child in childs if isinstance(child , FlowUnit) ])
        self.op = op
        self.label = label
        self.grad = 0.0
        self.backward = lambda : None
        
    def __repr__(self):
        return f"𝑓𝑙𝑜𝑤𝑈𝑛𝑖𝑡({self.data})"

    
    def __radd__(self , other):
        return self + other
    
    def __add__(self , other):
        other = other if isinstance(other , FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data + other.data , (self , other) , label="+")
        def backward():
            self.grad += 1.0 *  out.data
            other.grad += 1.0 *  out.data
        out.backward = backward
        return out
    
    def __rmul__(self , other):
        return self * other
    
    def __mul__(self , other):
        other = other if isinstance(other , FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data * other.data , (self,other) , label="*")
        
        def backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out.backward = backward
        return out
    
    def __sub__(self , other):
        other = other if isinstance(other , FlowUnit) else FlowUnit(other)
        if(self.data > other.data):
            out = FlowUnit(self.data - other.data , (self,other) , label="-")
        else:
            out = FlowUnit(other.data - self.data, (self,other) , label="-") 
        def backward():
            self.grad += 1.0 * other.data
            other.grad += -1.0 * self.data
        out.backward = backward
        return out
        
    def __truediv__(self , other):
        assert isinstance(other , (float , int)) , ValueError("Only Float and Integer will taken")
        out = FlowUnit(self.data / other.data , (self,other) , label="/")
        
        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (self.grad / (other.data) ** 2) * out.grad
        other.backward = backward
        return other
    
    def pow(self , other):
        assert isinstance(other , (float , int)) , ValueError("Only Float and Integer will taken")
        out = FlowUnit(self.data ** other , (self,other) , label=f"{self}**{other}")
        
        def backward():
            self.grad += ((other * self.data) ** (other - 1)) * out.grad
        out.backward = backward
        return out
    
    def exp(self):
        x = self.data
        out = FlowUnit(math.exp(x)  , (self , ) , label='exp')
        def backward():
            self.grad += out.data * out.grad
        out.backward = backward
        return out
    
    def sigmoid(self):
        x = self.data
        out = FlowUnit(1 / (1 + math.exp(-x)) , (self , ) , label=f'sigmoid{self}') 
        def backward():
            self.grad += (out.data * (1 - out.data)) * out.grad 
        out.backward = backward
        return out
    
    def tanh(self):
        if self.data > 10:
            t = 1.0
        elif self.data < -10:
            t = -1.0
        else:
            t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = FlowUnit(t , (self ,) , 'tanh')
        def backward():
            self.grad += (1 - t ** 2) * out.grad 
        out.backward = backward
        return out
    
    def relu(self):
        x = self.data
        out = FlowUnit(math.max(0 , x) , (self, ) , label='relu')
        def backward():
            if(self.data > 0):
                self.grad += 1 * out.grad
            else:
                 self.grad += 0 * out.grad
        out.backward = backward
        return out
    
    def leaky_relu(self):
        alpha = 0.01
        x = self.data
        if(x < 0):
            out = FlowUnit(alpha * x , (self , ), label='leaky_relu')
        else:
            out = FlowUnit(x, (self , ), label='leaky_relu')
        
        def backward():
            if(x < 0):
                self.grad += alpha * out.grad
            else:
                self.grad += x * out.grad
        out.backward = backward
        return out
            
            
    def softmax(self):
        z = self.data
        N = len(z)
        expo = []
        expo_sum = 0.0
        for i in range(N):
            expo.append(math.exp(z[i]))
            expo_sum += expo[i]
        
        self.a = [a / expo for a in expo]
        
        self.out = [FlowUnit(value) for value in self.a]
        
        def backward(target):
            
            grad_data = [a - t for a , t in zip(self.a , target)]
            
            for i , grad in enumerate(grad_data):
                self.out[i].grad = grad
        
        self.backward_func = backward
        
        return self.out
    
    def Backpropagation(self):
        topo = []
        vistied = set()
        
        def bulid_topo(v):
            if v not in vistied:
                vistied.add(v)
                for child in v._prev:
                    bulid_topo(child)
                topo.append(child)
        bulid_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node.backward()
                
    


            """import math

class FlowUnit:
    def __init__(self, data, childs=(), op="", label=""):
        self.data = data
        self._prev = set([child for child in childs if isinstance(child, FlowUnit)])
        self.op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"𝑓𝑙𝑜𝑤𝑈𝑛𝑖𝑡({self.data})"

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data + other.data, (self, other), label="+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data * other.data, (self, other), label="*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, FlowUnit) else FlowUnit(other)
        out = FlowUnit(self.data - other.data, (self, other), label="-")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        if isinstance(other, FlowUnit):
            out = FlowUnit(self.data / other.data, (self, other), label="/")
            
            def _backward():
                self.grad += (1 / other.data) * out.grad
                other.grad += (-self.data / (other.data ** 2)) * out.grad
            out._backward = _backward
            return out
        elif isinstance(other, (float, int)):
            out = FlowUnit(self.data / other, (self,), label="/")
            def _backward():
                self.grad += (1 / other) * out.grad
            out._backward = _backward
            return out
        else:
            raise ValueError("Only FlowUnit, Float, and Integer are valid operands for division.")

    def pow(self, other):
        assert isinstance(other, (float, int)), ValueError("Only Float and Integer are taken")
        out = FlowUnit(self.data ** other, (self,), label=f"pow{other}")
        
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = FlowUnit(math.exp(x), (self,), label='exp')
        
        def _backward():
            self.grad += out.data * out.grad  # derivative of exp(x) is exp(x)
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        sig = 1 / (1 + math.exp(-x))
        out = FlowUnit(sig, (self,), label='sigmoid')
        
        def _backward():
            self.grad += (sig * (1 - sig)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) if abs(x) <= 10 else (1.0 if x > 0 else -1.0)
        out = FlowUnit(t, (self,), label='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = FlowUnit(max(0, self.data), (self,), label='relu')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        out = FlowUnit(self.data if self.data > 0 else alpha * self.data, (self,), label='leaky_relu')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad
        out._backward = _backward
        return out


    def softmax(self):
        z = self.data
        N = len(z)
        expo = []
        expo_sum = 0.0
        for i in range(N):
            expo.append(math.exp(z[i]))
            expo_sum += expo[i]
        
        self.a = [a / expo_sum for a in expo]
        
        self.out = [FlowUnit(value) for value in self.a]
        
        def backward(target):
            
            grad_data = [a - t for a , t in zip(self.a , target)]
            
            for i , grad in enumerate(grad_data):
                self.out[i].grad = grad
        
        self.backward_func = backward
        
        return self.out
    

    def backward(self):
        # Build topological order of all nodes that flow into this one
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Zero gradients
        for v in topo:
            v.grad = 0.0
            
        # Set gradient of output to 1
        self.grad = 1.0
        
        # Backward pass: apply chain rule
        for node in reversed(topo):
            node._backward()

def test_all_operations():
    print("\nTesting all operations with backpropagation:")
    print("-" * 50)
    
    # Initialize values
    a = FlowUnit(3.0)
    b = FlowUnit(4.0)
    
    # Test addition
    c = a + b
    c.backward()
    print(f"Addition (a + b):")
    print(f"c = {c.data}, ∂c/∂a = {a.grad}, ∂c/∂b = {b.grad}")
    
    # Reset gradients
    a.grad = b.grad = 0.0
    
    # Test multiplication
    d = a * b
    d.backward()
    print(f"\nMultiplication (a * b):")
    print(f"d = {d.data}, ∂d/∂a = {a.grad}, ∂d/∂b = {b.grad}")
    
    # Test more complex expression
    expr = (a + b) * (a - b)
    expr.backward()
    print(f"\nComplex expression (a + b) * (a - b):")
    print(f"result = {expr.data}, ∂/∂a = {a.grad}, ∂/∂b = {b.grad}")
    
    # Test activation functions
    x = FlowUnit(2.0)
    
    # ReLU
    relu_out = x.relu()
    relu_out.backward()
    print(f"\nReLU(2.0):")
    print(f"output = {relu_out.data}, gradient = {x.grad}")
    
    # Sigmoid
    x.grad = 0.0  # Reset gradient
    sig_out = x.sigmoid()
    sig_out.backward()
    print(f"\nSigmoid(2.0):")
    print(f"output = {sig_out.data}, gradient = {x.grad}")
    
    # Softmax
    z = FlowUnit([2.0, 1.0, 0.1])
    z.target = [1, 0, 0]  # One-hot encoded target
    softmax_out = z.softmax()
    for out in softmax_out:
        out.backward()
    print(f"\nSoftmax([2.0, 1.0, 0.1]):")
    print(f"output = {[out.data for out in softmax_out]}")
    print(f"gradient = {z.grad}")
def check_softmax_and_sigmoid_gradients():
    print("\nChecking softmax and sigmoid gradients with PyTorch:")
    print("-" * 50)
    
    # Test case: Sigmoid
    x = FlowUnit(2.0)
    sigmoid_out = x.sigmoid()
    sigmoid_out.backward()

    x_torch = torch.tensor(2.0, requires_grad=True)
    sigmoid_torch = torch.sigmoid(x_torch)
    sigmoid_torch.backward()

    print(f"Sigmoid:")
    print(f"FlowUnit gradient -> ∂/∂x = {x.grad}")
    print(f"PyTorch gradient -> ∂/∂x = {x_torch.grad.item()}")

    # Test case: Softmax
    z = FlowUnit([2.0, 1.0, 0.1])
    softmax_out = z.softmax()
    z.target = [1, 0, 0]  # One-hot encoded target
    z.backward_func(z.target)

    z_torch = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    softmax_torch = torch.softmax(z_torch, dim=0)
    target_torch = torch.tensor([1.0, 0.0, 0.0])  # One-hot target
    loss_torch = torch.sum((softmax_torch - target_torch) ** 2)  # L2 loss
    loss_torch.backward()

    print(f"\nSoftmax:")
    print(f"FlowUnit output -> {[out.data for out in softmax_out]}")
    print(f"PyTorch output -> {softmax_torch.detach().tolist()}")
    print(f"FlowUnit gradient -> {z.out[0].grad}, {z.out[1].grad}, {z.out[2].grad}")
    print(f"PyTorch gradient -> {z_torch.grad.tolist()}")

if __name__ == "__main__":
    check_softmax_and_sigmoid_gradients()

            """