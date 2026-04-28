import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1], nn.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2], nn.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3], nn.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4], nn.GELU())),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5], nn.GELU())),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3 ,3, 1]
    sample_input = torch.tensor([[1., 0., -1]])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

    print_gradients(model_without_shortcut, sample_input)
    """
    layers.0.0.weight has gradient mean of 0.001531339017674327
    layers.1.0.weight has gradient mean of 0.0008734675939194858
    layers.2.0.weight has gradient mean of 0.002111606765538454
    layers.3.0.weight has gradient mean of 0.0030934528913348913
    layers.4.0.weight has gradient mean of 0.00788064580410719
    """

    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_without_shortcut, sample_input)
    """
    layers.0.0.weight has gradient mean of 0.24866615235805511
    layers.1.0.weight has gradient mean of 0.8006525039672852
    layers.2.0.weight has gradient mean of 0.38361987471580505
    layers.3.0.weight has gradient mean of 0.3954206109046936
    layers.4.0.weight has gradient mean of 1.001085877418518
    """
