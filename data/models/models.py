from torch import nn
import torch.nn.functional as F

class Mnist1(nn.Module):

    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super(Mnist1, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l_out(x)
        return F.log_softmax(x)

class Mnist2(nn.Module):

    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super(Mnist2, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l_out(x)
        return F.log_softmax(x)

class Mnist3(nn.Module):

    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super(Mnist3, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l_out(x)
        return F.log_softmax(x)

class Mnist4(nn.Module):

    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super(Mnist4, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l_out(x)
        return F.log_softmax(x)

class Mnist5(nn.Module):

    def __init__(self, input_size=784, hidden_size=200, output_size=10):
        super(Mnist5, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        x = self.relu(x)
        x = self.l_out(x)
        return F.log_softmax(x)
    
class FF_10_20_10(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FF_10_20_10, self).__init__()
        self.layer1 = nn.Linear(input_size, 10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

