import math
import torch
from torch import nn
import torch.distributed as dist


def init_linear_like_pytorch(linear_layer: nn.Linear):
    """
    Initializes a Linear layer the same way PyTorch does by default.
    
    Args:
        linear_layer (nn.Linear): The linear layer to initialize.
    """
    if not isinstance(linear_layer, nn.Linear):
        raise TypeError("Expected an instance of nn.Linear.")

    # Weight: Kaiming uniform
    nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
    
    # Bias: Uniform in [-bound, bound] where bound = 1 / sqrt(fan_in)
    if linear_layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_layer.bias, -bound, bound)
        
        
class RunningMeanTracker(nn.Module):
    def __init__(self, dim, momentum=0.1):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(dim))
        self.momentum = momentum
        self.initialized = False
    
    def update(self, batch: torch.Tensor):
        # batch: (B, D)
        batch = batch.clone().detach()
        local_mean = batch.mean(dim=0)

        # sync across all processes
        world_size = dist.get_world_size()
        if world_size > 1:
            mean_clone = local_mean
            dist.all_reduce(mean_clone, op=dist.ReduceOp.SUM)
            global_mean = mean_clone / world_size
        else:
            global_mean = local_mean
            
        # ensure dtype match
        global_mean = global_mean.to(self.running_mean.dtype)

        if not self.initialized:
            self.running_mean.copy_(global_mean)
            self.initialized = True
        else:
            self.running_mean.lerp_(global_mean, self.momentum)

    def center(self, batch):
        return batch - self.running_mean
    
    
class ConfigDict(dict):
    """Dictionary subclass that allows both dot notation and bracket notation access."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")