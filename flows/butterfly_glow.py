 

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.nn import Parameter

from  .flow import Flow
from  .actnorm import ActNorm2dFlow
from  .butter_layer import *
from  .conv import Conv1x1Flow
from  .nice import NICE
from .utils import squeeze2d, unsqueeze2d, split2d, unsplit2d

import copy


class Prior(Flow):
    """
    prior for multi-scale architecture
    """
    def __init__(self, in_channels, hidden_channels=None, s_channels=None, scale=True, inverse=False, factor=2, use_conv1by1=False,
                 use_intermediate_perm=False, permute_channel=False,
                 weight_init_types=None, reverse_perm=False, bi_direction=False,
                 butterfly_partitions=None, LU_decomposed=False, rotate=False):
        super(Prior, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.use_conv1by1 = use_conv1by1
        if use_conv1by1:
            self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse, LU_decomposed=LU_decomposed)
        else:
            self.conv1x1 = None

        if weight_init_types is not None and use_intermediate_perm:

            if butterfly_partitions is not None:
                self.butterfly = general_butterfly_block_partition(
                    layer_nums=[len(wit) for wit in weight_init_types],
                    partitions=butterfly_partitions, inverse=self.inverse, share_weights=False, use_bias=False,
                    weight_init_types=weight_init_types, reverse_perm=reverse_perm,bi_direction=bi_direction,
                    rotate=rotate,permute_channel=permute_channel
                )
            else:
                self.butterfly = general_butterfly_block_simple(layer_num=len(weight_init_types), data_dim=3 * 32 * 32,
                                                              inverse=self.inverse,
                                                              share_weights=False, use_bias=False,
                                                              weight_init_types=weight_init_types,
                                                              model_layers = len(weight_init_types),
                                                                reverse_perm=reverse_perm,
                                                                bi_direction=bi_direction,permute_channel=permute_channel)
        else:
            self.butterfly = None
        self.nice = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, inverse=inverse, factor=factor)
        self.z1_channels = self.nice.z1_channels

        self.module_dict = nn.ModuleDict({
            'actnorm': self.actnorm,
            'conv1x1': self.conv1x1,
            'butterfly': self.butterfly,
            'nice': self.nice
        })

    def sync(self):
        # pass
        if self.conv1x1 is not None:
            self.conv1x1.sync() # todo: maybe remove this function (could cause bugs, check sync for butterfly flow)
        if self.butterfly is not None:
            self.butterfly.sync() # todo: maybe remove this function (could cause bugs, check sync for butterfly flow)

    def get_butterfly(self):
        return nn.ModuleDict({'butterfly': self.butterfly})


    def assign_module(self, target_module_dict):
        '''
        target_module_dict: dict containing replacement key and module
        '''
        if 'butterfly' in target_module_dict:
            self.butterfly = target_module_dict['butterfly']
    def get_parameters(self):

        model_params = list(self.actnorm.parameters()) + list(self.nice.parameters())
        if self.conv1x1 is not None:
            model_params += list(self.conv1x1.parameters())

        if self.butterfly is not None:
            butterfly_params = list(self.butterfly.parameters())
        else:
            butterfly_params = []

        return model_params, butterfly_params
    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)


        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.forward(out)
            logdet_accum = logdet_accum + logdet

        # added

        if self.butterfly is not None:
            out, logdet = self.butterfly.forward(out)
            logdet_accum = logdet_accum + logdet

        # added
        out, logdet = self.nice.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.nice.backward(input, s=s)

        # added
        if self.butterfly is not None:
            out, logdet = self.butterfly.backward(out)
            logdet_accum = logdet_accum + logdet

        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.backward(out)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        # added

        if self.butterfly is not None:
            out, logdet = self.butterfly.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet


        out, logdet = self.nice.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class ButterGlowStep(Flow):
    """
    A step of Glow. A Conv1x1 followed with a NICE
    """
    def __init__(self, in_channels, hidden_channels=512, s_channels=0, scale=True, inverse=False,
                 coupling_type='conv', slice=None, heads=1, pos_enc=True, dropout=0.0, use_conv1by1=False,
                 use_intermediate_perm=False, weight_init_types=None, reverse_perm=False, bi_direction=False,
                 butterfly_partitions=None, permute_channel=False,
                 LU_decomposed=False, rotate=False):
        super(ButterGlowStep, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.use_conv1by1 = use_conv1by1
        if use_conv1by1:
            self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse, LU_decomposed=LU_decomposed)
        else:
            self.conv1x1 = None
        if weight_init_types is not None and use_intermediate_perm:

            if butterfly_partitions is not None:
                self.butterfly = general_butterfly_block_partition(
                    layer_nums=[len(wit) for wit in weight_init_types],
                    partitions=butterfly_partitions, inverse=self.inverse, share_weights=False, use_bias=False,
                    weight_init_types=weight_init_types,
                    bi_direction=bi_direction, reverse_perm=reverse_perm, rotate=rotate, permute_channel=permute_channel
                )
            else:
                self.butterfly = general_butterfly_block_simple(layer_num=len(weight_init_types), data_dim=3 * 32 * 32,
                                                                inverse=self.inverse,
                                                                share_weights=False, use_bias=False,
                                                                weight_init_types=weight_init_types,
                                                                model_layers=len(weight_init_types),
                                                                reverse_perm=reverse_perm,
                                                                bi_direction=bi_direction, permute_channel=permute_channel)

        else:
            self.butterfly = None
        self.coupling = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels,
                             scale=scale, inverse=inverse, type=coupling_type, slice=slice, heads=heads, pos_enc=pos_enc, dropout=dropout)

        # self.module_dict = nn.ModuleDict({
        #     'actnorm': self.actnorm,
        #     'conv1x1': self.conv1x1,
        #     'butterfly': self.butterfly,
        #     'coupling': self.coupling
        # })
    def sync(self):
        if self.conv1x1 is not None:
            self.conv1x1.sync()
        if self.butterfly is not None:
            self.butterfly.sync() # todo: maybe remove this function (could cause bugs, check sync for butterfly flow)

    def get_butterfly(self):
        return nn.ModuleDict({'butterfly': self.butterfly})

    def assign_module(self, target_module_dict):
        '''
        target_module_dict: dict containing replacement key and module
        '''

        if 'butterfly' in target_module_dict:
            self.butterfly = target_module_dict['butterfly']

    def get_parameters(self):

        model_params = list(self.actnorm.parameters()) + list(self.coupling.parameters())
        if self.conv1x1 is not None:
            model_params += list(self.conv1x1.parameters())

        if self.butterfly is not None:
            butterfly_params = list(self.butterfly.parameters())
        else:
            butterfly_params = []

        return model_params, butterfly_params
    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)


        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.forward(out)
            logdet_accum = logdet_accum + logdet
        # added

        if self.butterfly is not None:
            out, logdet = self.butterfly.forward(out)
            logdet_accum = logdet_accum + logdet

        # added
        out, logdet = self.coupling.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.coupling.backward(input, s=s)

        if self.butterfly is not None:
            out, logdet = self.butterfly.backward(out)
            logdet_accum = logdet_accum + logdet


        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.backward(out)
            logdet_accum = logdet_accum + logdet
        # added
        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        if self.conv1x1 is not None:
            out, logdet = self.conv1x1.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        # # added

        if self.butterfly is not None:
            out, logdet = self.butterfly.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet


        out, logdet = self.coupling.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def visualize_weights(self):

        if self.butterfly is None:
            return None
        else:
            imgs = self.butterfly.visualize_weights()
            return imgs

class ButterGlowTopBlock(Flow):
    """
    ButterGlow Block (squeeze at beginning)
    """
    def __init__(self, num_steps, in_channels, scale=True, inverse=False, use_conv1by1=False,
                 use_intermediate_perm=False, permute_channel=False, hidden_channels=None,
                 weight_init_types=None, reverse_perm=False,bi_direction=False,
                 butterfly_partitions=None,  LU_decomposed=False, rotate=False):
        super(ButterGlowTopBlock, self).__init__(inverse)

        if hidden_channels is None:
            glowstep_hidden = 512
        else:
            glowstep_hidden = hidden_channels
        steps = [ButterGlowStep(in_channels, scale=scale, inverse=inverse, use_conv1by1=use_conv1by1,
                                use_intermediate_perm=use_intermediate_perm, weight_init_types=weight_init_types,
                                reverse_perm=reverse_perm,bi_direction=bi_direction, hidden_channels=glowstep_hidden,
                                LU_decomposed=LU_decomposed, butterfly_partitions=butterfly_partitions, rotate=rotate,
                                permute_channel=permute_channel) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

        self.module_dict = nn.ModuleDict(
            {'steps': self.steps}
        )

    def sync(self):
        for step in self.steps:
            step.sync()

    def get_butterfly(self):
        butterflys = nn.ModuleList([module.get_butterfly() for module in self.steps])

        return nn.ModuleDict({'steps':butterflys})

    def assign_module(self, target_module_dict):
        '''
        target_module_dict: dict containing replacement key and module
        '''
        assert len(self.steps) == len(target_module_dict['steps'])
        for i, val in enumerate(target_module_dict['steps']):
            self.steps[i].assign_module(val)

    def get_parameters(self):
        model_params, butterfly_params = [],[]
        for module in self.steps:
            mp, bp = module.get_parameters()
            model_params += mp
            butterfly_params += bp

        return model_params, butterfly_params

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

    def visualize_weights(self):

        ret = []
        for step in self.steps:
            ret.append(step.visualize_weights())

        return ret

class ButterGlowInternalBlock(Flow):
    """
    ButterGlow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_steps, in_channels, scale=True, inverse=False, use_conv1by1=False,
                 use_intermediate_perm=False, permute_channel=False, hidden_channels=None,
                 weight_init_types=None, reverse_perm=False,bi_direction=False,
                 butterfly_partitions=None, LU_decomposed=False, rotate=False):
        super(ButterGlowInternalBlock, self).__init__(inverse)
        if hidden_channels is None:
            glowstep_hidden = 512
            prior_hidden = None
        else:
            glowstep_hidden = prior_hidden = hidden_channels
        steps = [ButterGlowStep(in_channels, scale=scale, inverse=inverse, use_conv1by1=use_conv1by1,
                                use_intermediate_perm=use_intermediate_perm,hidden_channels=glowstep_hidden,
                                weight_init_types=weight_init_types, reverse_perm=reverse_perm,bi_direction=bi_direction,
                                butterfly_partitions=butterfly_partitions, LU_decomposed=LU_decomposed, rotate=rotate, permute_channel=permute_channel) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        self.prior = Prior(in_channels, scale=scale, inverse=True,  use_conv1by1=use_conv1by1,
                           use_intermediate_perm=use_intermediate_perm, hidden_channels=prior_hidden,
                           weight_init_types=weight_init_types, reverse_perm=reverse_perm,bi_direction=bi_direction,
                           butterfly_partitions=butterfly_partitions,
                           LU_decomposed=LU_decomposed, rotate=rotate, permute_channel=permute_channel)

        self.module_dict = nn.ModuleDict(
            {'steps': self.steps,
             'prior': self.prior}
        )

    def sync(self):
        for step in self.steps:
            step.sync()
        self.prior.sync()

    def get_butterfly(self):
        butterflys = nn.ModuleList([module.get_butterfly() for module in self.steps])

        prior_bf = self.prior.get_butterfly()

        return nn.ModuleDict({'steps': butterflys, 'prior': prior_bf})

    def assign_module(self, target_module_dict):
        '''
        target_module_dict: dict containing replacement key and module
        '''
        assert len(self.steps) == len(target_module_dict['steps'])
        for i, val in enumerate(target_module_dict['steps']):
            self.steps[i].assign_module(val)

        self.prior.assign_module(target_module_dict['prior'])

    def get_parameters(self):
        model_params, butterfly_params = [], []
        for module in self.steps:
            mp, bp = module.get_parameters()
            model_params += mp
            butterfly_params += bp
        mp, bp = self.prior.get_parameters()
        model_params += mp
        butterfly_params += bp
        return model_params, butterfly_params
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

    def visualize_weights(self):

        ret = []
        for step in self.steps:
            ret.append(step.visualize_weights())

        return ret

class Butter_Glow(Flow):
    """
    Butter_Glow
    """
    def __init__(self, levels, num_steps, factors, in_channels, scale=True, inverse=False,
                 use_conv1by1=False, hidden_channels=None,
                 use_intermediate_perm=False, LU_decomposed=False, permute_channel=False,
                 weight_init_types=None, model_layers=None, reverse_perm=False, bi_direction=False,
                 butterfly_partitions=None, rotate=False):
        super(Butter_Glow, self).__init__(inverse)
        assert levels > 1, 'Butter_Glow should have at least 2 levels.'
        assert levels == len(num_steps)
        blocks = []
        self.levels = levels
        self.factors = factors
        self.use_conv1by1 = use_conv1by1

        if weight_init_types is not None and not use_intermediate_perm:

            if butterfly_partitions is not None:
                self.butterfly = general_butterfly_block_partition(
                    layer_nums=[len(wit) for wit in weight_init_types],
                    partitions=butterfly_partitions, inverse=self.inverse, share_weights=False, use_bias=False,
                    weight_init_types=weight_init_types, reverse_perm=reverse_perm, rotate=rotate,
                    bi_direction=bi_direction, permute_channel=permute_channel
                )
            else:
                self.butterfly = general_butterfly_block_simple(layer_num=len(weight_init_types), data_dim=3 * 32 * 32,
                                                              inverse=self.inverse,
                                                              share_weights=False, use_bias=False,
                                                              weight_init_types=weight_init_types,
                                                              model_layers = model_layers,
                                                                reverse_perm = reverse_perm,
                                                                bi_direction=bi_direction, permute_channel=permute_channel)
        else:
            self.butterfly = None

        self.inverse = inverse

        for level in range(levels):
            if level == levels - 1:
                in_channels = in_channels * (self.factors[level]**2)
                macow_block = ButterGlowTopBlock(num_steps[level], in_channels, scale=scale, inverse=inverse, use_conv1by1=use_conv1by1,  LU_decomposed=LU_decomposed,
                                                 use_intermediate_perm=use_intermediate_perm, weight_init_types=weight_init_types, reverse_perm=reverse_perm,
                                                 bi_direction=bi_direction, hidden_channels=hidden_channels,
                                                 butterfly_partitions=butterfly_partitions, rotate=rotate, permute_channel=permute_channel)
                blocks.append(macow_block)
            else:
                in_channels = in_channels * (self.factors[level]**2)
                macow_block = ButterGlowInternalBlock(num_steps[level], in_channels, scale=scale, inverse=inverse,  use_conv1by1=use_conv1by1, LU_decomposed=LU_decomposed,
                                                      use_intermediate_perm=use_intermediate_perm, weight_init_types=weight_init_types, reverse_perm=reverse_perm,
                                                      bi_direction=bi_direction,hidden_channels=hidden_channels,
                                                      butterfly_partitions=butterfly_partitions, rotate=rotate, permute_channel=permute_channel)
                blocks.append(macow_block)
                in_channels = in_channels // self.factors[level]

                if weight_init_types is not None and len(weight_init_types) > 0:
                
                    if len(weight_init_types)> 0 and (isinstance(weight_init_types[0], list) or isinstance(weight_init_types[0], tuple) ):
                        weight_init_types = copy.deepcopy(weight_init_types)
                        for k in range(len(weight_init_types)):
                            weight_init_types[k].pop()
                        butterfly_partitions = copy.copy(butterfly_partitions)
                        butterfly_partitions = [bp//self.factors[level] for bp in butterfly_partitions]
                    else:
                
                        weight_init_types = copy.copy(weight_init_types)
                        weight_init_types.pop()
               



        self.blocks = nn.ModuleList(blocks)

        self.module_dict = nn.ModuleDict({
            'blocks': self.blocks,
            'butterfly': self.butterfly
        })

    def sync(self):
        for block in self.blocks:
            block.sync()

    def get_butterfly(self):
        butterflys = {'blocks': nn.ModuleList([module.get_butterfly() for module in self.blocks]),
                      'butterfly': self.butterfly}

        return nn.ModuleDict(butterflys)

    def assign_module(self, target_module_dict):
        '''
        target_module_dict: dict containing replacement key and module
        '''

        assert len(self.blocks) == len(target_module_dict['blocks'])
        for i, val in enumerate(target_module_dict['blocks']):
            self.blocks[i].assign_module(val)

        if 'butterfly' in target_module_dict:
            self.butterfly = target_module_dict['butterfly']

    def get_parameters(self):
        model_params, butterfly_params = [], []
        for module in self.blocks:
            mp, bp = module.get_parameters()
            model_params += mp
            butterfly_params += bp

        if self.butterfly is not None:
            butterfly_params += list(self.butterfly.parameters())
        return model_params, butterfly_params
    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []

        if self.butterfly is not None:
            out, logdet = self.butterfly(out)
            logdet_accum = logdet_accum + logdet

        for i, block in enumerate(self.blocks):
            out = squeeze2d(out, factor=self.factors[i])
            out, logdet = block.forward(out)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, ButterGlowInternalBlock):
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
            if isinstance(block, ButterGlowInternalBlock):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
            out, logdet = block.backward(out)
            logdet_accum = logdet_accum + logdet
            out = unsqueeze2d(out, factor=self.factors[i])

        if self.butterfly is not None:
            out, logdet = self.butterfly.backward(out)
            logdet_accum = logdet_accum + logdet
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:

        logdet_accum = data.new_zeros(data.size(0))
        out = data
        if self.butterfly is not None:
            out, logdet = self.butterfly(out)

            logdet_accum = logdet_accum + logdet
        outputs = []
        for i, block in enumerate(self.blocks):
            out = squeeze2d(out, factor=self.factors[i])
            out, logdet = block.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, ButterGlowInternalBlock):
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
    def from_params(cls, params: Dict) -> "Butter_Glow":

        return Butter_Glow(**params)

    def visualize_weights(self):
        viz = []
        for module in self.blocks:
            imgs = module.visualize_weights()
            viz.append(imgs)

        return viz

Butter_Glow.register('butter_glow')
