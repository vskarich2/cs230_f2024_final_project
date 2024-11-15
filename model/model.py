import torch.nn as nn
from types import SimpleNamespace


class Attention(nn.Module):
    def __init__(self, config):
        pass
        #TODO

    def forward(self, x):
        pass
        #TODO


class CrossAttention(nn.Module):
    def __init__(self, config):
        pass
        # TODO

    def forward(self, q, k, v):
        pass
        # TODO


class GPT2FullyConnected(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO

    def forward(self, x):
        pass
        # TODO


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO

    def forward(self, x, enc_out):
        pass
        # TODO

class GPT2ModelWithVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO

    def forward(self, image, input_ids, labels=None):
        pass
        # TODO

if __name__ == '__main__':
    model_config = SimpleNamespace()
    model = GPT2ModelWithVision.from_pretrained(model_config)