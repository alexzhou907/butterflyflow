 

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.nn import Parameter

from  .flow import Flow
from  .actnorm import ActNorm2dFlow
from  .conv import Conv1x1Flow
from  .nice import NICE
from .utils import squeeze2d, unsqueeze2d, split2d, unsplit2d


class Prior(Flow):
    """
    prior for multi-scale architecture
    """
    def __init__(self, in_channels, hidden_channels=None, s_channels=None, scale=True, inverse=False, factor=2, LU_decomposed=False):
        super(Prior, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse, LU_decomposed=LU_decomposed)
        self.nice = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, inverse=inverse, factor=factor)
        self.z1_channels = self.nice.z1_channels

    def sync(self):
        self.conv1x1.sync()

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)

        out, logdet = self.conv1x1.forward(out)
        # print(logdet.shape)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.nice.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.nice.backward(input, s=s)

        out, logdet = self.conv1x1.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        out, logdet = self.conv1x1.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.nice.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class GlowStep(Flow):
    """
    A step of Glow. A Conv1x1 followed with a NICE
    """
    def __init__(self, in_channels, hidden_channels=512, s_channels=0, scale=True, inverse=False,
                 coupling_type='conv', slice=None, heads=1, pos_enc=True, dropout=0.0, LU_decomposed=False):
        super(GlowStep, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse, LU_decomposed=LU_decomposed)
        self.coupling = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels,
                             scale=scale, inverse=inverse, type=coupling_type, slice=slice, heads=heads, pos_enc=pos_enc, dropout=dropout)

    def sync(self):
        self.conv1x1.sync()

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)

        out, logdet = self.conv1x1.forward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.coupling.backward(input, s=s)

        out, logdet = self.conv1x1.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        out, logdet = self.conv1x1.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class GlowTopBlock(Flow):
    """
    Glow Block (squeeze at beginning)
    """
    def __init__(self, num_steps, in_channels, scale=True, inverse=False, LU_decomposed=False):
        super(GlowTopBlock, self).__init__(inverse)
        steps = [GlowStep(in_channels, scale=scale, inverse=inverse, LU_decomposed=LU_decomposed) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    def sync(self):
        for step in self.steps:
            step.sync()

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class GlowInternalBlock(Flow):
    """
    Glow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_steps, in_channels, scale=True, inverse=False, LU_decomposed=False):
        super(GlowInternalBlock, self).__init__(inverse)
        steps = [GlowStep(in_channels, scale=scale, inverse=inverse, LU_decomposed=LU_decomposed) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        self.prior = Prior(in_channels, scale=scale, inverse=True, LU_decomposed=LU_decomposed)

    def sync(self):
        for step in self.steps:
            step.sync()
        self.prior.sync()

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.prior.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        out, logdet_accum = self.prior.backward(input)
        for step in reversed(self.steps):
            out, logdet = step.backward(out)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.prior.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class Glow(Flow):
    """
    Glow
    """
    def __init__(self, levels, num_steps, factors, in_channels, scale=True, inverse=False, LU_decomposed=False,**kwargs):
        super(Glow, self).__init__(inverse)
        assert levels > 1, 'Glow should have at least 2 levels.'
        assert levels == len(num_steps) == len(factors)
        blocks = []
        self.factors = factors
        self.levels = levels
        for level in range(levels):
            if level == levels - 1:
                in_channels = in_channels * (factors[level]**2)
                macow_block = GlowTopBlock(num_steps[level], in_channels, scale=scale, inverse=inverse, LU_decomposed=LU_decomposed)
                blocks.append(macow_block)
            else:
                in_channels = in_channels * (factors[level]**2)
                macow_block = GlowInternalBlock(num_steps[level], in_channels, scale=scale, inverse=inverse, LU_decomposed=LU_decomposed)
                blocks.append(macow_block)
                in_channels = in_channels // factors[level]
        self.blocks = nn.ModuleList(blocks)

    def sync(self):
        for block in self.blocks:
            block.sync()

    def get_parameters(self):
        return list(self.parameters()), []

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            out = squeeze2d(out, factor=self.factors[i])
            out, logdet = block.forward(out)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, GlowInternalBlock):
                out1, out2 = split2d(out, out.size(1) // self.factors[i])
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=self.factors[-1])
        for j in reversed(range(self.levels - 1)):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=self.factors[j])
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        out = squeeze2d(input, factor=self.factors[-1])
        for j in range(self.levels - 1):
            out1, out2 = split2d(out, out.size(1) // self.factors[j])
            outputs.append(out2)
            out = squeeze2d(out1, factor=self.factors[j])

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if isinstance(block, GlowInternalBlock):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
            out, logdet = block.backward(out)
            logdet_accum = logdet_accum + logdet
            out = unsqueeze2d(out, factor=self.factors[i])
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            out = squeeze2d(out, factor=self.factors[i])
            out, logdet = block.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, GlowInternalBlock):
                out1, out2 = split2d(out, out.size(1) // self.factors[i])
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=self.factors[-1])
        for j in reversed(range(self.levels - 1)):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=self.factors[j])
        assert len(outputs) == 0
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "Glow":
        return Glow(**params)

    def visualize_weights(self, *args):
        pass

Glow.register('glow')
