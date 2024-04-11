import torch.nn as nn
import torch
import math

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)
    
def get_activation_fn(name: str):
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")

class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
        
class ProteinResNetConfig(object):
    def __init__(self,
                 hidden_size: int = 512,
                 output_hidden_states: bool = False,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, 
                 first_dilated_layer = 2, 
                 resnet_bottleneck_factor = 0.5,
                 dilation_rate = 3, 
                 nblocks = 5):
        self.hidden_size = hidden_size
        self.output_hidden_states = output_hidden_states
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.first_dilated_layer = first_dilated_layer
        self.resnet_bottleneck_factor = resnet_bottleneck_factor
        self.nblocks = nblocks
        self.dilation_rate = dilation_rate
        
class MaskedConv1d(nn.Conv1d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)

class ProteinResNetLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ProteinResNetBlock(nn.Module):

    def __init__(self, config, layer_index):
        super().__init__()
        shifted_layer_index = layer_index - config.first_dilated_layer + 1
        dilation_rate = max(1, config.dilation_rate**shifted_layer_index)
        
        num_bottleneck_units = math.floor(
            config.resnet_bottleneck_factor * config.hidden_size)
        
        self.conv1 = MaskedConv1d(
            config.hidden_size, num_bottleneck_units, 9, 
            dilation=dilation_rate, padding=0)
        self.bn1 = nn.BatchNorm1d(config.hidden_size)
        self.conv2 = MaskedConv1d(
            num_bottleneck_units, config.hidden_size, 1, dilation=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_bottleneck_units)
        self.activation_fn = get_activation_fn(config.hidden_act)

    def forward(self, x, input_mask=None):
        identity = x

        out = self.bn1(x)
        out = self.activation_fn(out)
        out = self.conv1(out, input_mask)

        out = self.bn2(out)
        out = self.activation_fn(out)
        out = self.conv2(out, input_mask)

        #out += identity

        return out

class ResNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.nblocks = config.nblocks
        self.layer = nn.ModuleList(
            [ProteinResNetBlock(config, ind) for ind in range(config.nblocks)])

    def forward(self, hidden_states, input_mask=None):
        all_hidden_states = ()
        for layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states, input_mask)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs