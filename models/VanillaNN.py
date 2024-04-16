from torch import nn


class VanillaNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=None,
        flatten_input=False,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.flatten_input = flatten_input
        self.flatten_layer = nn.Flatten()

        if not hidden_sizes or hidden_sizes is None:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
            )
        else:
            layers = []
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(activation())

            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                layers.append(activation())

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_input:
            x = self.flatten_layer(x)

        x = self.layers(x)
        return x
