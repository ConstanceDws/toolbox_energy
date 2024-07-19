import torch
import torch.nn as nn
from deepspeed.profiling.flops_profiler import get_model_profile
from thop import profile


class DynamicMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DynamicMLP, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])

        for _ in range(2, num_layers + 1):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.layers.extend([nn.Linear(hidden_size, output_size), nn.Softmax(dim=1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)  

        for layer in self.layers:
            x = layer(x)

        return x

class DynamicCNN(nn.Module):
    def __init__(self, input_channels, output_classes, hidden_size, num_layers, num_frame):
        super(DynamicCNN, self).__init__()

        layers = []
        in_channels = input_channels

        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            in_channels = hidden_size

        self.conv_layers = nn.ModuleList(layers)

        fc_input_size = hidden_size * (128 // (2 ** num_layers)) * (num_frame // (2 ** num_layers))

        self.fc = nn.Linear(fc_input_size, output_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.softmax(x)
        return x


class DynamicRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, num_directions=1):
        super(DynamicRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=(num_directions == 2))

        fc_input_size = hidden_size * num_directions
        self.fc = nn.Linear(fc_input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1) 

        x = x.view(batch_size, sequence_length, -1)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = out.contiguous().view(batch_size, -1)

        out = self.softmax(self.fc(out))

        return out
    
class DynamicCRNN(nn.Module):
    def __init__(self, input_channels, output_classes, hidden_size, num_layers):
        super(DynamicCRNN, self).__init__()

        layers = []
        in_channels = input_channels
        hidden_size_cnn, hidden_size_rnn = hidden_size
        num_layers_cnn, num_layers_rnn = num_layers
        for _ in range(num_layers_cnn):
            layers.extend([
                nn.Conv2d(in_channels, hidden_size_cnn, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            in_channels = hidden_size_cnn

        self.conv_layers = nn.ModuleList(layers)
        self.rnn = nn.GRU(input_size=hidden_size_cnn, hidden_size=hidden_size_rnn, num_layers=num_layers_rnn, batch_first=True)
        self.fc = nn.Linear(hidden_size_rnn, output_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1, x.size(1))
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = 'mlp' 
    num_frame = 64
    hidden_size = 128
    num_layers = 1

    if model == 'mlp':
        model = DynamicMLP(input_size= 128*num_frame, hidden_size=hidden_size, output_size=10, num_layers=num_layers)

    elif model == 'cnn':
        model = DynamicCNN(input_channels = 1, output_classes= 10, hidden_size=hidden_size, num_layers=num_layers, num_frame=num_frame)

    elif model == 'rnn':
        model = DynamicRNN(input_size = 128*num_frame, output_size=10, hidden_size=hidden_size, num_layers=num_layers)

    elif model == 'crnn':
        model = DynamicCRNN(input_channels = 1, output_classes=10, hidden_size=hidden_size, num_layers=num_layers)

    dummy_input = torch.rand((1, 1, 128, num_frame))
    macs, params = profile(model = model, inputs=(dummy_input, ))
    print(f'MACS: {macs} PARAM : {params}')

    shape = (1, 1, 128, num_frame)
    flops, macs, params = get_model_profile(model=model, input_shape=(shape), module_depth=-1)
    print(flops,macs,params)




