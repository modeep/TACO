from pytorch.nn import *
from collections import OrderedDict

class prenet(Module):
    def __init__(self, input):
        super(prenet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5))
        ]))

    def forward(input):
        output = self.net(input)

        return output
