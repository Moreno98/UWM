import torch.nn as nn
import torch
import torch.nn.functional as F

class CustomLinear(nn.Linear):
    def __init__(self, 
                module: object,
                name: str,
                in_features: int, 
                out_features: int, 
                bias: bool = True,
                device=None, 
                dtype=None
        ):
        self.module = module
        self.name = name
        super(CustomLinear, self).__init__(in_features, out_features, bias)
        self.device = device
        self.dtype = dtype
        self.mask = torch.ones(out_features, in_features, device=device, dtype=dtype)

    def __str__(self):
        return self.module, self.name

    @staticmethod
    def from_pretrained(module, name, linear: nn.Linear):
        linear_ = CustomLinear(
            module,
            name,
            linear.in_features, 
            linear.out_features, 
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype
        )
        linear_.weight = linear.weight
        linear_.bias = linear.bias
        return linear_

    def apply_mask(self):
        return self.weight * self.mask

    def update_mask(self, mask):
        self.mask = mask.to(self.weight.dtype).to(self.weight.device)

    def forward(self, input):
        return F.linear(input, self.apply_mask(), self.bias)