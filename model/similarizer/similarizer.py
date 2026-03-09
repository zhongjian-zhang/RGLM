import torch.nn as nn


class Similarizer(nn.Module):
    def __init__(self, input_dim, output_dim, depth):
        super().__init__()
        modules = [nn.Linear(input_dim, output_dim)]
        if depth > 2:
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
        self.net = nn.Sequential(*modules)
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming for hidden Linear (with GELU), small-normal for final layer."""
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for lin in linear_layers:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if len(linear_layers) > 0:
            last = linear_layers[-1]
            nn.init.normal_(last.weight, mean=0.0, std=0.02)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        
    def forward(self, x):
        return self.net(x)
