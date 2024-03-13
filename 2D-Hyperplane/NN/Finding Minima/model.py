import torch
from torch.nn import Module
from torch import nn
import math

class InhiLinear(Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        '''
        #scale = torch.Tensor(size_out)
        scale = torch.zeros(size_in)
        scale = scale + 0.5
        scale = torch.bernoulli(scale)
        scale = scale * 2 - 1
        #print(scale)
        self.scale = scale
        #self.scale = nn.Parameter(scale)
        '''
        scale = torch.Tensor(size_in)
        #self.scale = nn.Parameter(scale)

        #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.weights, 0, 10)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bound = 0
        nn.init.uniform_(self.bias, -bound, bound)
        #nn.init.uniform_(self.bias, 0, 1)

        #nn.init.uniform_(self.scale, -1, 1)

        self.size_in = size_in

    def forward(self, x):
        #print(self.bias.size())
        #print(x.size())
        #print(torch.mul(self.size_in, torch.softmax(self.weights, dim = 1)))
        #mid = torch.mm(x, torch.mul(self.size_in, torch.softmax(self.weights, dim = 1)).t())
        #print(nn.functional.relu(self.weights))
        #mid = torch.mm(x, nn.functional.relu(self.weights).t())
        mid = torch.mm(x, nn.functional.softmax(self.weights, dim = 1).t())
        #mid = torch.mm(x, torch.sigmoid(self.weights).t())
        #print("weight:", torch.sigmoid(self.weights))
        #print("weight:" , nn.functional.softmax(self.weights, dim = 1))
        #mid = torch.mm(x, nn.functional.normalize(self.weights, p=2.0, dim=1, eps=1e-12).t())
        #weight0 = torch.mul(self.weights, 1/torch.sum(self.weights, 1, True))
        #print(torch.sum(weight0, 1, True))
        #mid = torch.mm(x, weight0.t())
        #print(mid.size())
        #mid1 = torch.add(mid, self.bias)
        #return nn.functional.relu(mid1)
        #return torch.mul(self.scale, nn.functional.relu(mid1))
        #mid = torch.mm(x, self.weights.t())
        #print("bias:" , - nn.functional.relu(self.bias))
        return torch.add(mid, - nn.functional.relu(self.bias))
        #return torch.add(mid, self.bias)

class InhiScale(Module):
    def __init__(self, size):
        super().__init__()
        scale = torch.Tensor(size)
        self.scale = nn.Parameter(scale)
        #nn.init.uniform_(self.scale, -1, 1)
        nn.init.uniform_(self.scale, - math.sqrt(6), math.sqrt(6))
        #nn.init.normal_(self.scale, 0, math.sqrt(2))

    def forward(self, x):
        #print("scale:" , self.scale)
        return torch.mul(self.scale, x)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(3, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, 2)
        '''
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(2, 2)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(2, 2)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(2, 2)
        '''

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)
        '''
        y = self.relu3(y)
        y = self.fc4(y)
        y = self.relu4(y)
        y = self.fc5(y)
        y = self.relu5(y)
        y = self.fc6(y)
        '''

        return y
