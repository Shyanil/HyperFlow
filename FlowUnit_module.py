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
        x = self.data
        out = FlowUnit(math.exp(x) - math.exp(-x) / math.exp(x) + math.exp(-x)  , (self, ) , label='tanh')
        def backward():
            self.grad += (1 - out ** 2) * out.grad 
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
        sum_exp = 0.0
        for k in range(N):
            expo.append(math.exp(z[k]))
            sum_exp += expo[k]
 
        a = []
        for i in range(N):
            a.append(expo[i] / sum_exp)
        
        out = []
        
        for i in range(N):
            out.append(FlowUnit(a[i]))
        
        
        def backward():
            for i in range(len(z)):
               for j in range(len(a)):
                   if(i == j):
                       out[j].grad += a[j] * (1 - a[j])
                   else:
                       out[j].grad += - (a[j] * a[i])
        for unit in out:
            unit.backward = backward
        return out               
            
        
     
            
            
             