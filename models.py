import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(             # input shape (1,10000,12)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=5,  # n_filters
                kernel_size=(200, 3),  # filter size
                stride=(50, 1),  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
        )
        self.conv2 = nn.Sequential(               # input shape (5,99,7)
            nn.Conv2d(5, 10, (20, 2), (4, 1), 1),  # output shape
            nn.ReLU(),                             # activation
            nn.MaxPool2d(kernel_size=2),                      # output shape (10,10,4)
        )
        self.out = nn.Linear(10 * 10 * 4, 6)      # fully connected layer, output 6 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        feature = x
        output = self.out(x)
        return feature, output


