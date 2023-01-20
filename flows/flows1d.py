import math
import types

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


def squeeze1d(x, factor=2) -> torch.Tensor:
    assert factor >= 1
    if factor == 1:
        return x
    batch, time_len, n_channels = x.size()
    assert time_len % factor == 0
    # [batch,time_len, channels,] -> [batch,time_len/factor,factor* channels]
    x = x.reshape(-1, time_len // factor, factor* n_channels)

    # [batch, channels, factor, factor, height/factor, width/factor]
    # x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    # [batch, channels*factor*factor, height/factor, width/factor]
    # x = x.view(-1, n_channels * factor * factor, height // factor, width // factor)
    return x


def unsqueeze1d(x: torch.Tensor, factor=2) -> torch.Tensor:
    factor = int(factor)
    assert factor >= 1
    if factor == 1:
        return x
    batch, time_len, n_channels = x.size()
    num_bins = factor
    assert n_channels >= num_bins and n_channels % num_bins == 0

    # [batch, channels, height, width] -> [batch, factor, factor, channels/(factor*factor), height, width]
    x = x.reshape(-1, time_len*factor,  n_channels // num_bins).contiguous()


    # [batch, channels, height, width] -> [batch, channels/(factor*factor), factor, factor, height, width]
    # x = x.view(-1, n_channels // num_bins, factor, factor, height, width)
    # [batch, channels/(factor*factor), height, factor, width, factor]
    # x = x.permute(0, 1, 4, 2, 5, 3).contiguous()

    return x


def split1d(x: torch.Tensor, z1_channels):
    z1 = x[..., :z1_channels]
    z2 = x[..., z1_channels:]
    return z1, z2


def unsplit1d(xs) -> torch.Tensor:
    # [batch, time_len, channels]
    return torch.cat(xs, dim=-1)



class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 pre_exp_tanh=False):
        super(MADESplit, self).__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs,
                              mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs,
                               mask_type='output')

        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.s_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))

        act_func = activations[t_act]
        self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.t_trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs,
                                                   output_mask))
        
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)
            
            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)
            
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1)

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.in_channels= num_inputs
        self.weight = nn.Parameter(torch.ones(1, num_inputs))
        self.bias = nn.Parameter(torch.zeros(1, num_inputs))
        self.initialized = False

        nn.init.normal_(self.weight, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            out = inputs  * torch.exp(self.weight) + self.bias
            out = out.view(-1,self.in_channels)
            mean = out.mean(dim=0).view(1,self.in_channels)
            std = out.std(dim=0).view(1,self.in_channels)
            inv_stdv = 1. / (std + 1e-6)

            self.weight.data.add_(inv_stdv.log())
            self.bias.data.add_(-mean).mul_(inv_stdv)
            self.initialized = True

        batch, T, channels = inputs.size()
        if mode == 'direct':
            return inputs  * torch.exp(self.weight) + self.bias, \
                   self.weight.sum( -1).repeat(inputs.size(0)).mul(T)
        else:
            return (inputs- self.bias) * torch.exp(
                -self.weight) ,\
            self.weight.sum(-1).repeat(inputs.size(0)).mul(T)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0))
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0))


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).repeat(inputs.size(0))
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).repeat(
                    inputs.size(0))


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.register_buffer("perm", torch.randperm(num_inputs))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        # self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]


        self.z1_channels = num_inputs // 2
        self.scale_net = nn.Sequential(
            nn.Linear(self.z1_channels, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, self.z1_channels))
        self.translate_net = nn.Sequential(
            nn.Linear(self.z1_channels, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, self.z1_channels))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        # mask = self.mask

        masked_inputs = inputs #* mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        z1 = masked_inputs[..., :self.z1_channels]
        z2 = masked_inputs[..., self.z1_channels:]
        if mode == 'direct':
            log_s = self.scale_net(z1) #* (1 - mask)
            t = self.translate_net(z1) #* (1 - mask)
            s = torch.exp(log_s)
            z2 = z2 * s + t
            return torch.cat([z1,z2], dim=-1), log_s.reshape(inputs.size(0),-1).sum(-1)
        else:
            log_s = self.scale_net(z1) #* (1 - mask)
            t = self.translate_net(z1) #* (1 - mask)
            s = torch.exp(-log_s)
            z2 = (z2 - t) * s
            return torch.cat([z1,z2], dim=-1), -log_s.reshape(inputs.size(0),-1).sum(-1)




class FlowSequentialStacked(nn.Module):
    def __init__(self, args, modules:nn.ModuleDict):
        super().__init__()
        self.blocks = modules
        self.args = args

    def forward(self, input: torch.Tensor, cond_inputs=None, mode='direct'):

        logdet_accum = input.new_zeros(input.size(0))

        out = input
        if mode == 'direct':
            outputs = []

            for level, block in enumerate(self.blocks):
                out = squeeze1d(out, factor=2)
                for module in block.values():

                    out, logdet = module.forward(out, cond_inputs=cond_inputs, mode=mode)
                    logdet_accum = logdet_accum + logdet
                if level < self.args.levels-1:
                    out1, out2 = split1d(out, out.size(-1) // 2)
                    outputs.append(out2)
                    out = out1


            out = unsqueeze1d(out, factor=2)
            for j in reversed(range(self.args.levels - 1)):
                out2 = outputs.pop()
                out = unsqueeze1d(unsplit1d([out, out2]), factor=2)
            assert len(outputs) == 0
            return out, logdet_accum
        else:
            outputs = []
            out = squeeze1d(input, factor=2)
            for j in range(self.args.levels - 1):
                out1, out2 = split1d(out, out.size(-1) // 2)
                outputs.append(out2)
                out = squeeze1d(out1, factor=2)

            for i in reversed(range(len(self.blocks))):
                block = self.blocks[i]
                if i < self.args.levels-1:
                    out2 = outputs.pop()
                    out = unsplit1d([out, out2])

                for module in reversed(block.values()):

                    out, logdet = module.forward(out, cond_inputs=cond_inputs,mode=mode)

                    logdet_accum = logdet_accum + logdet
                out = unsqueeze1d(out, factor=2)

            assert len(outputs) == 0

            return out, logdet_accum

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).reshape(inputs.shape[0], -1).sum(-1)
        # print(log_probs.shape, log_jacob.shape)
        return (log_probs + log_jacob), {'log_prior':log_probs, 'log_jacob':log_jacob}


    def get_butterfly(self):
        ret = nn.ModuleDict()
        for internal_block in self.blocks:
            for key, values in internal_block.items():
                if key.startswith('butterfly'):
                    ret.update({key: values})

        return ret

    def assign_module(self, target_dict):

        for internal_block in self.blocks:
            for key, values in target_dict.items():
                if key in internal_block:
                    internal_block[key] = values


    def get_parameters(self):
        model_params, butterfly_params = [], []
        for internal_block in self.blocks:
            for key, values in internal_block.items():
                if key.startswith('butterfly'):
                    butterfly_params += list(values.parameters())
                else:
                    model_params += list(values.parameters())


        return model_params, butterfly_params


class FlowSequential(nn.Module):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """
    def __init__(self, modules:nn.ModuleDict):
        super().__init__()
        self.layers = modules

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self.layers.values():

                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self.layers.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        # print(log_probs.shape, log_jacob.shape)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

    def get_butterfly(self):
        ret = nn.ModuleDict()
        for key, values in self.layers.items():
            if key.startswith('butterfly'):
                ret.update({key: values})

        return ret

    def assign_module(self, target_dict):
        for key, values in target_dict.items():
            self.layers[key] = values


    def get_parameters(self):
        model_params, butterfly_params = [], []
        for key, values in self.layers.items():
            if key.startswith('butterfly'):
                butterfly_params += list(values.parameters())
            else:
                model_params += list(values.parameters())

        return model_params, butterfly_params

def elu_derivative(x, slope=1.0):
    slope1 = torch.ones_like(x)
    x = torch.min(x, torch.ones_like(x) * 70.)
    slope2 = torch.exp(x) * slope
    return torch.where(x > 0, slope1, slope2)


def elu_inverse(x, slope=1.0):
    slope2 = torch.log(torch.clamp(x / slope + 1., min=1e-10)) # x<0
    return torch.where(x > 0, x, slope2)

def sigmoid_derivative(sig_x):
    return sig_x * (1- sig_x)


def rotation_init(x):
    # x1=1, x2=3
    theta = torch.rand((x.shape[0], x.shape[2])) * 2 * np.pi
    x[:,0, :,0] = torch.cos(theta)
    x[:,0, :,1] = -torch.sin(theta)
    x[:,1, :,0] = torch.sin(theta)
    x[:,1, :,1] = torch.cos(theta)
    return x


def identity_init(x):
    zeros = torch.zeros((x.shape[0], x.shape[2]))
    ones = torch.ones((x.shape[0], x.shape[2]))

    x[:, 0, :, 0] = ones
    x[:, 0, :, 1] = zeros
    x[:, 1, :, 0] = zeros
    x[:, 1, :, 1] = ones
    return x

class butterfly_block(nn.Module): # nn.Module
    def __init__(self, data_dim=2, level=0, use_act=True, share_weights=False, use_bias=False, weight_init_type='iden'):
        super(butterfly_block, self).__init__()
        self.use_act = use_act
        self.data_dim = data_dim
        self.level = level
        self.share_weights = share_weights
        assert data_dim % (2**(level + 1)) == 0
        self.l = data_dim // (2**level) # todo: need to change the data_dim to be intialized based on data! since the input dimension is changing

        if self.share_weights:
            ll = 1

        else:
            ll = self.l // 2

        if weight_init_type == 'iden':
            # self.w1 = torch.randn(2 ** level, 2, 1, 2)
            self.w1 = identity_init(  # rotation_init( #
                # self.w1 = rotation_init(
                torch.randn(2 ** level, 2, ll, 2))
        elif weight_init_type == 'rot':
            self.w1 = rotation_init(
                torch.randn(2 ** level, 2, ll, 2))
        elif weight_init_type == 'orth':

            self.w1 = torch.randn(2 ** level, 2, ll, 2) #TODO: try different permutation
            torch.nn.init.orthogonal_(self.w1, 1)

        else:
            raise NotImplementedError
        self.w1 = nn.Parameter(self.w1)


        self.use_bias = use_bias
        if use_bias:
            self.b1 = nn.Parameter(
                torch.zeros(1, self.data_dim) # the weights of b might also need to be shared
            )

        self.act =  F.elu
        self.d_act = elu_derivative
        self.inverse_activation = elu_inverse


    def init_conv_weight(self, weight):
        init.xavier_normal_(weight, gain=0.1)

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def logit(self, x):
        return torch.log(x) - torch.log1p(-x)

    def forward(self, x):
        # x needs to be a power of 2 for the general algorithm
        # log_det: batch
        # x: (batch, data_dim)
        batch = x.shape[0]

        x = x.reshape(batch, -1)
        w1 = self.w1
        x = x.reshape(batch, 2**self.level, 2, self.l // 2)  # batch, 2^level, 2, dim/2^(level+1) (in the image case, need to check how the partition works)
        x = x[..., None] * w1[None, ...]  # batch, 2^level, 2, dim/2^(level+1), 2
        x = torch.sum(x, dim=2)  # batch, 2^level, dim/2^(level+1), 2

        # solve potential ordering issue, fix idenitity bug (there should be a permutation, otherwise even identity initialization will not give an indetity output)
        x = x.permute(0, 1, 3, 2) # batch, 2^level, 2, dim/2^(level+1) fixed bug: added this line

        if self.use_act:
            if self.use_bias:
                x = x + self.b1.reshape(1, *x[0, ...].shape)  # batch, 2^level, dim/2^(level+1), 2
            x = self.act(x)  # batch, dim, 1
            det = w1[None, ...] * self.d_act( x[..., None])  # batch, 2**level, 2, self.l//2, 2 (element-wise multiplication)

            log_det = torch.sum(torch.log(torch.abs(det[:, :, 0, :, 0] * det[:, :, 1, :, 1] - det[:, :, 1, :, 0] * det[:, :, 0, :, 1] )), dim=(1, 2))
            # det([[A,B],[C,D]])=det(A)*det(D-CA^{-1}B)
            # log_det: batch
            x = x.reshape(batch, -1)  # batch, data_dim
            return x, log_det  # (batch, dim), batch
        else:
            log_det = torch.log(torch.abs(w1[:,0,:,0])).sum() + torch.log(torch.abs(w1[:,1,:,1] - w1[:,1,:,0] * w1[:,0,:,1] /  w1[:,0,:,0])).sum()  # det([[A,B],[C,D]])=det(A)*det(D-CA^{-1}B)

            if self.share_weights:
                log_det = log_det * (self.l // 2)

            x = x.reshape(batch, -1)  # batch, data_dim
            if self.use_bias:
                x = x + self.b1  # batch, data_dim
            return x, log_det * torch.ones(batch).to(x.device)  # (batch, dim), batch

    def backward(self, y):
        y_shape = y.shape  # batch, dim
        batch = y.shape[0]
        y = y.reshape(batch, -1)
        # b1 = self.b1 * 0. # remove multiply by 0. (looks like weights are hard to train, might consider sharing or removing it, 1x1 conv does not have the bias?)
        with torch.no_grad():
            if self.use_act:
                y = self.inverse_activation(y)

            if self.use_bias:
                y = y - self.b1

            y = y.reshape(batch, 2**self.level, 2, self.l//2)

            w = self.w1
            w_inv = torch.zeros_like(w)
            A = w[:, 0, :, 0]
            B = w[:, 1, :, 0]
            C = w[:, 0, :, 1]
            D = w[:, 1, :, 1]
            w_inv[:, 0, :, 0] = -D / (B * C - D * A)  # 1./(A-B*C/D)
            w_inv[:, 1, :, 0] = -B / (A * D - B * C)  # -B /(A*D-B*C)
            w_inv[:, 0, :, 1] = C / (B * C - D * A)  # -C / (A*D-B*C)
            w_inv[:, 1, :, 1] = A / (
                        A * D - B * C)  # 1./D + C*B/(D*(A*D-B*C)) # inverse of block matrix [[A,B],[C,D]], A,B,C,D all digonal matrices

            y = y[..., None] * w_inv[None, ...]  # batch, 2^level, 2, dim/2^(level+1), 2
            y = torch.sum(y, dim=2)  # batch, 2^level, dim/2^(level+1), 2

        y = y.permute(0, 1, 3, 2) # remove this line after fix the permutation bug in forward
        y = y.reshape(*y_shape)
        return y


class general_butterfly_block_simple(nn.Module):  # nn.Module
    # todo: understand the issue why in glow, the data_dim is changing in different blocks!
    def __init__(self, data_dim, layer_num=1, share_weights=False, use_bias=False,
                 weight_init_types=None,  reverse_perm=False,bi_direction=False,):
        super(general_butterfly_block_simple, self).__init__()
        self.data_dim = data_dim
        self.layer_num = layer_num
        # self.init_flag = False
        self.share_weights = share_weights
        self.bi_direction = bi_direction

        self.use_bias = use_bias
        self.reverse_perm = reverse_perm

        if weight_init_types is None:
            self.weight_init_types = ['iden' for _ in range(layer_num)]
        else:
            self.weight_init_types = weight_init_types


        # if not self.init_flag:
        #     self.init_flag = True
        # data_dim = np.prod(x.shape[1:])

        model_list = [
            butterfly_block(data_dim=self.data_dim, level=i, use_act=False, share_weights=self.share_weights,
                            use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]) for
            i in range(self.layer_num)]
        if bi_direction:
            model_list += [
                butterfly_block(data_dim=self.data_dim, level=i, use_act=False, share_weights=self.share_weights,
                                use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]) for
                i in reversed(range(self.layer_num))]
        if self.reverse_perm:
            model_list = list(model_list)
        else:
            model_list = list(reversed(model_list))

        self.model = nn.ModuleList(model_list)
        # print(len(self.model), data_dim)

    def forward(self, x, cond_inputs=None, mode='direct'):
        input_shape = x.shape

        log_det = torch.zeros(input_shape[0], device=x.device)
        if mode == 'direct':
            for i, layer in enumerate(self.model):
                x, log_det_x = layer(x)
                log_det = log_det + log_det_x  # todo: change the shape of log_det from scalr to batch (but seems that glow also output scalar)
        else:
            for i, layer in reversed(list(enumerate(self.model))):
                x = layer.backward(x)

        x = x.reshape(*input_shape)
        log_det = log_det.reshape(log_det.shape[0])
        return x, log_det

    # def backward(self, y):
    #     input_shape = y.shape
    #     for i, layer in reversed(list(enumerate(self.model))):
    #         y = layer.backward(y)
    #
    #     y = y.reshape(*input_shape)
    #     return y, 0.  # out, logdet (0.) not used during sampling
    #



class general_butterfly_block_ordered(nn.Module): #nn.Module
    # todo: understand the issue why in glow, the data_dim is changing in different blocks!
    def __init__(self, data_dim, layer_num=1, share_weights=False, use_bias=False,
                 weight_init_types=None, reverse_perm=False, bi_direction=False, ):
        super(general_butterfly_block_ordered, self).__init__()
        self.data_dim = data_dim
        self.layer_num = layer_num
        # self.init_flag = False
        self.share_weights = share_weights
        self.bi_direction = bi_direction

        self.use_bias = use_bias
        self.reverse_perm = reverse_perm

        if weight_init_types is None:
            self.weight_init_types = ['iden' for _ in range(layer_num)]
        else:
            self.weight_init_types = weight_init_types

        # if not self.init_flag:
        #     self.init_flag = True
        # data_dim = np.prod(x.shape[1:])

        model_list = [
            butterfly_block(data_dim=self.data_dim, level=0, use_act=False, share_weights=self.share_weights,
                            use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]) for
            i in range(self.layer_num)]
        # if bi_direction:
        #     model_list += [
        #         butterfly_block(data_dim=self.data_dim, level=i, use_act=False, share_weights=self.share_weights,
        #                         use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]) for
        #         i in reversed(range(self.layer_num))]
        if self.reverse_perm:
            model_list = list(model_list)
        else:
            model_list = list(reversed(model_list))

        self.model = nn.ModuleList(model_list)
        # print(len(self.model), data_dim)

    def forward(self, x,cond_inputs=None, mode='direct'):
        input_shape = x.shape
        level = 32 // x.shape[1]

        log_det = torch.zeros(input_shape[0], device=x.device)
        if mode == 'direct':
            for i, layer in enumerate(self.model):
                x = forward_order(x.reshape(*input_shape), level=i) # added
                x, log_det_x = layer(x)
                log_det = log_det + log_det_x # todo: change the shape of log_det from scalr to batch (but seems that glow also output scalar)
                x = backward_order(x.reshape(*input_shape), level=i) # added
        else:
            for i, layer in reversed(list(enumerate(self.model))):
                x = forward_order(x.reshape(*input_shape), level=i)  # added
                x = layer.backward(x)
                x = backward_order(x.reshape(*input_shape), level=i)  # added

        x = x.reshape(*input_shape)
        return x, log_det


def forward_order(x, level=0):
    assert level >= 0
    batch, h, c = x.shape

    # x = x.flatten(start_dim=1)  # batch, dim
    x = x.reshape(batch, -1, 2, 2 ** level, c)
    x = x.permute(0, 2, 1,3,4) # corrected
    x = x.reshape(batch, h,c)

    return x


def backward_order(x, level=0):
    assert level >= 0
    batch, h, c = x.shape
    x = x.reshape(batch, 2, -1, 2 ** level, c) # corrected
    x = x.permute(0, 2, 1, 3, 4) # corrected
    x = x.reshape(batch, h, c)

    return x