

import torch
import numpy as np

import math
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.utils

from  .flow import Flow
from overrides import overrides
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import io
import copy

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

def perm_init(x):
  # x1=1, x2=3

  sel = torch.randint(2, size=x[...,0,:,0].shape)
  x[..., 0, :, 0] = sel
  x[..., 0, :, 1] = 1-sel
  x[..., 1, :, 0] = 1-sel
  x[..., 1, :, 1] = sel
  return x

class butterfly_block(Flow): # nn.Module
    def __init__(self, data_dim=2, level=0, use_act=True, inverse=False, share_weights=False, use_bias=False, weight_init_type='iden'):
        super(butterfly_block, self).__init__(inverse)
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

            self.w1 = torch.randn(2 ** level * ll, 4) #TODO: try different permutation
            torch.nn.init.orthogonal_(self.w1, 1)
            self.w1 = self.w1.reshape(2 ** level, ll, 2,2).transpose(1,2)

        else:
            raise NotImplementedError
        self.w1 = nn.Parameter(self.w1)

        # self.w1 = nn.Parameter(
        #     torch.randn(2**level, 2, self. l//2, 2) * 0.5 # the last 2 dimension represents the non-zero element in the butterfly block
        # )
        # self.w1 = nn.Parameter(
        #     torch.ones(2 ** level, 2, self.l // 2, 2) / 2. + torch.randn(2**level, 2, self. l//2, 2) # the last 2 dimension represents the non-zero element in the butterfly block
        # )
        # self.b1 = 0.
        self.use_bias = use_bias
        if use_bias:
            self.b1 = nn.Parameter(
                torch.zeros(1, self.data_dim) # the weights of b might also need to be shared
            )
            # self.init_conv_bias(self.w1, self.b1)

        # self.init_conv_weight(self.w1)
        # nn.init.orthogonal_(self.w1) #todo: initinalization is very important! #looks like this init works worse then without any extra init
        # self.init_conv_bias(self.w1, self.b1)

        self.act = torch.sigmoid # F.elu
        self.d_act = sigmoid_derivative # elu_derivative
        self.inverse_activation = self.logit


    def init_conv_weight(self, weight):
        init.xavier_normal_(weight, gain=0.1)

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def logit(self, x):
        return torch.log(x) - torch.log1p(-x)

    @overrides
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
            det = w1[None, ...] * x[..., None] * (
                        1 - x[..., None])  # batch, 2**level, 2, self.l//2, 2 (element-wise multiplication)

            log_det = torch.sum(torch.log(torch.abs(det[:, :, 0, :, 0] * det[:, :, 1, :, 1] - det[:, :, 1, :, 0] * det[:, :, 0, :, 1] )), dim=(1, 2))
            # det([[A,B],[C,D]])=det(A)*det(D-CA^{-1}B)
            # log_det: batch
            x = x.reshape(batch, -1)  # batch, data_dim
            return x, log_det  # (batch, dim), batch
        else:
            log_det = torch.log(torch.abs(w1[:, 0, :, 0] * w1[:, 1, :, 1] - w1[:, 1, :, 0] * w1[:, 0, :, 1])).sum()  # det([[A,B],[C,D]])=det(A)*det(D-CA^{-1}B)

            if self.share_weights:
                log_det = log_det * (self.l // 2)

            x = x.reshape(batch, -1)  # batch, data_dim
            if self.use_bias:
                x = x + self.b1  # batch, data_dim
            return x, log_det * torch.ones(batch).to(x.device)  # (batch, dim), batch

    @overrides
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

            # y = y.reshape(batch, 2 ** self.level, self.l // 2, 2)
            # y = y.permute(0, 1, 3, 2)  # batch, 2^level, 2, dim/2^(level+1) (in the image case, need to check how the partition works)
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



def forward_order(x, level=1, flip=False):
    assert level > 0
    batch, c, h, w = x.shape

    if flip:
        x = x.permute(0, 1, 3, 2)
    # x = x.flatten(start_dim=1)  # batch, dim
    x = x.reshape(batch, c, -1, 2, 2 ** (level - 1))
    x = x.permute(0, 3, 1, 2, 4) # corrected
    x = x.reshape(batch, c, h, w)

    return x


def backward_order(x, level=1, flip=False):
    assert level > 0
    batch, c, h, w = x.shape
    x = x.reshape(batch, 2, c, -1, 2 ** (level - 1)) # corrected
    x = x.permute(0, 2, 3, 1, 4) # corrected
    x = x.reshape(batch, c, h, w)

    if flip:
        x = x.permute(0, 1, 3, 2).contiguous()

    return x


class general_butterfly_block(Flow): #nn.Module
    # todo: understand the issue why in glow, the data_dim is changing in different blocks!
    def __init__(self, layer_num=2, data_dim=2, inverse=False, v2=False, share_weights=False, use_bias=False, weight_init=None):
        super(general_butterfly_block, self).__init__(inverse)
        self.layer_num = layer_num
        self.init_flag = False
        self.v2 = v2
        self.share_weights = share_weights

        self.use_bias = use_bias

        if weight_init is None:
            self.weight_init = [None for _ in range(layer_num)]
    @overrides
    def forward(self, x):
        input_shape = x.shape
        level = 32 // x.shape[1]
        # x = x.reshape(x.shape[0], -1)  # batch, dim

        if not self.init_flag:
            self.init_flag = True
            data_dim = np.prod(x.shape[1:])
            if not self.v2:
                model_list = [butterfly_block(data_dim=data_dim, level=0, use_act=False, share_weights=self.share_weights, use_bias=self.use_bias).to(x.device) for
                              i in range(2 * self.layer_num - 1)]
                model_list.append(
                    butterfly_block(data_dim=data_dim, level=0, use_act=False, share_weights=self.share_weights, use_bias=self.use_bias).to(x.device))
            else:
                raise NotImplementedError

            self.model = nn.ModuleList(model_list)

        log_det = 0.
        for i, layer in enumerate(self.model):
            x = forward_order(x.reshape(*input_shape), level=i // 2 + 1, flip=((i + 1) % 2 == 0)) # added
            x, log_det_x = layer(x)
            log_det = log_det + log_det_x # todo: change the shape of log_det from scalr to batch (but seems that glow also output scalar)
            x = backward_order(x.reshape(*input_shape), level=i // 2 + 1, flip=((i + 1) % 2 == 0)) # added
        x = x.reshape(*input_shape)
        return x, log_det

    @overrides
    def backward(self, y):
        input_shape = y.shape
        for i, layer in reversed(list(enumerate(self.model))):
            y = forward_order(y, level=i // 2 + 1, flip=((i + 1) % 2 == 0)) # added
            y = layer.backward(y)
            y = backward_order(y, level=i//2+1, flip=((i+1)%2==0)) # added

        y = y.reshape(*input_shape)
        return y, 0. #out, logdet (0.) not used during sampling

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)


    def sync(self): # todo: check the right way of implementing sync
        pass
        # self.weight_inv.copy_(self.weight.data.inverse()) # this is the one in 1x1 conv


class general_butterfly_block_simple(Flow):  # nn.Module
    # todo: understand the issue why in glow, the data_dim is changing in different blocks!
    def __init__(self, layer_num=2, data_dim=2, inverse=False, v2=False, share_weights=False, use_bias=False,
                 weight_init_types=None, model_layers=None, reverse_perm=False, bi_direction=False, permute_channel=False):
        super(general_butterfly_block_simple, self).__init__(inverse)
        self.layer_num = layer_num
        self.init_flag = False
        self.v2 = v2
        self.share_weights = share_weights

        self.permute_channel= permute_channel


        self.use_bias = use_bias
        self.reverse_perm = reverse_perm

        self.bi_direction = bi_direction

        if weight_init_types is None:
            self.weight_init = ['iden' for _ in range(layer_num)]
        else:
            self.weight_init_types = weight_init_types

        self.model_layers = model_layers
        if self.model_layers is None:
            self.model_layers = len(self.weight_init_types)
    @overrides
    def forward(self, x):
        if self.permute_channel:
            x = x.permute(0,2,3,1).contiguous()
        input_shape = x.shape
        level = 32 // x.shape[1]
        # x = x.reshape(x.shape[0], -1)  # batch, dim

        if not self.init_flag:
            self.init_flag = True
            data_dim = np.prod(x.shape[1:])

            if not self.v2:
                model_list = [
                    butterfly_block(data_dim=data_dim, level=i, use_act=False, share_weights=self.share_weights,
                                    use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]).to(x.device) for
                    i in range(self.layer_num)]
                if self.bi_direction:
                    model_list += [
                        butterfly_block(data_dim=data_dim, level=i, use_act=False, share_weights=self.share_weights,
                                        use_bias=self.use_bias, weight_init_type=self.weight_init_types[i]).to(x.device) for
                        i in reversed(range(self.layer_num))]
                if self.reverse_perm:
                    model_list = list(model_list)
                else:
                    model_list = list(reversed(model_list))
            else:
                raise NotImplementedError

            # self.model = nn.ModuleList(model_list[:len([l for l in reversed(self.weight_init) if l is not None])])

            self.model = nn.ModuleList(model_list[:self.model_layers])

        log_det = 0.
        for i, layer in enumerate(self.model):
            x, log_det_x = layer(x)
            log_det = log_det + log_det_x  # todo: change the shape of log_det from scalr to batch (but seems that glow also output scalar)

        x = x.reshape(*input_shape)
        if self.permute_channel:
            x = x.permute(0,3,1,2).contiguous()
        return x, log_det

    @overrides
    def backward(self, y):
        if self.permute_channel:
            y = y.permute(0,2,3,1).contiguous()
        input_shape = y.shape
        for i, layer in reversed(list(enumerate(self.model))):
            y = layer.backward(y)

        y = y.reshape(*input_shape)
        if self.permute_channel:
            y = y.permute(0,3,1,2).contiguous()
        return y, 0.  # out, logdet (0.) not used during sampling

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    def sync(self):  # todo: check the right way of implementing sync
        pass
        # self.weight_inv.copy_(self.weight.data.inverse()) # this is the one in 1x1 conv

    def visualize_weights(self):

        ret = []
        for i, layer in enumerate(self.model):
            w1 = layer.w1
            if w1.shape[-2] == 1:
                w1 = w1.expand(-1,-1,layer.l//2, -1)
            level = w1.shape[0]

            ww = torch.zeros(level, 2, 2, layer.l//2, layer.l//2, device=w1.device)
            ww[:,:,:,torch.arange(layer.l//2),torch.arange(layer.l//2)] = w1.transpose(-2,-1)
            www = torch.cat( [torch.cat([ww[:,0,0], ww[:,0,1]], dim=1),torch.cat([ww[:,1,0], ww[:,1,1]], dim=1) ], dim=-1)
            www = www[:10]
            fig_size = 32#min(20 // 6 * www.shape[-1], 32)
            fig, axs = plt.subplots(len(www), figsize=(fig_size, fig_size*len(www)))

            if len(www) == 1:
                axs = [axs]
            for j, d in enumerate(www):
                im = axs[j].imshow(d.detach().cpu().numpy(), cmap='viridis',interpolation='none')
                cbar=fig.colorbar(im,ax=axs[j])
                cbar.ax.tick_params(labelsize=40)

            # fig.savefig(os.path.join(dir, 'weight_level_%03d.png'%i))
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            im = data.reshape((int(h), int(w), -1))
            plt.close()

            ret.append(im)

        return ret



class general_butterfly_block_partition(Flow):  # nn.Module

    def __init__(self, layer_nums=[], partitions=[], inverse=False, v2=False, share_weights=False, use_bias=False,
                 weight_init_types=None, reverse_perm=False, bi_direction=False, rotate=False, permute_channel=False):
        super(general_butterfly_block_partition, self).__init__(inverse)
        assert len(layer_nums) == len(partitions)
        self.layer_nums = layer_nums
        self.partitions = partitions
        self.init_flag = False
        self.v2 = v2
        self.share_weights = share_weights
        self.permute_channel = permute_channel

        self.use_bias = use_bias
        self.reverse_perm = reverse_perm
        self.bi_direction = bi_direction

        self.rotate = rotate

        if weight_init_types is None:
            self.weight_init_types = [['iden' for _ in range(ln)] for ln in layer_nums]
        else:
            self.weight_init_types = weight_init_types

        if self.rotate:
            assert self.weight_init_types is not None
            self.partition_list = [self.partitions[-i:] + self.partitions[:-i] for i in range(len(self.partitions))]
            self.layer_nums_list = [self.layer_nums[-i:] + self.layer_nums[:-i] for i in range(len(self.layer_nums))]
            self.weight_init_types_list = [self.weight_init_types[-i:] + self.weight_init_types[:-i] for i in range(len(self.weight_init_types))]
        else:
            self.partition_list = [self.partitions]
            self.layer_nums_list = [self.layer_nums]
            self.weight_init_types_list = [self.weight_init_types]


    @overrides
    def forward(self, x):
        if self.permute_channel:
            x = x.permute(0,2,3,1).contiguous()
        input_shape = x.shape

        if not self.init_flag:
            self.init_flag = True
            data_dim = np.prod(x.shape[1:])


            models_all_part = []
            for curr_partition, curr_layer_nums,curr_wit in zip(self.partition_list,self.layer_nums_list,self.weight_init_types_list):#rotate partition to get better mixture

                model_list = [[
                    butterfly_block(data_dim=partition_dim, level=i, use_act=False, share_weights=self.share_weights,
                                    use_bias=self.use_bias, weight_init_type=wit[i]).to(x.device) for
                    i in range(ln)] for ln, partition_dim, wit in zip(curr_layer_nums, curr_partition, curr_wit)]

                if self.bi_direction:
                    for k in range(len(model_list)):
                        model_list[k] += [
                            butterfly_block(data_dim=curr_partition[k], level=i, use_act=False, share_weights=self.share_weights,
                                            use_bias=self.use_bias, weight_init_type=curr_wit[k][i]).to(x.device) for
                            i in reversed(range(curr_layer_nums[k]))]
                if self.reverse_perm:
                    model_list = [list(ml) for ml in model_list]
                else:
                    model_list = [list(reversed(ml)) for ml in model_list]

                models_all_part.append(nn.ModuleList([nn.ModuleList(ml) for ml in model_list]))


            # self.model = nn.ModuleList(model_list[:len([l for l in reversed(self.weight_init) if l is not None])])

            self.model = nn.ModuleList(models_all_part)
            print(self.partitions, data_dim)

        log_det = 0.
        x = x.reshape(input_shape[0], -1)
        assert x.shape[-1] == sum(self.partitions), 'dimension: {}, partition: {}'.format(x.shape[-1], self.partitions)

        for curr_partition, curr_part_model in zip(self.partition_list, self.model):
            x_partition = list(torch.split(x, curr_partition, dim=-1))
            for p in range(len(x_partition)):
                for i, layer in enumerate(curr_part_model[p]):
                    x_partition[p], log_det_x = layer(x_partition[p])
                    log_det = log_det + log_det_x  # todo: change the shape of log_det from scalr to batch (but seems that glow also output scalar)

            x = torch.cat(x_partition, dim=-1)
        x = x.reshape(*input_shape)

        if self.permute_channel:
            x = x.permute(0,3,1,2).contiguous()
        return x, log_det

    @overrides
    def backward(self, y):
        if self.permute_channel:
            y = y.permute(0,2,3,1).contiguous()
        input_shape = y.shape
        y = y.reshape(input_shape[0], -1)
        for curr_partition, curr_part_model in reversed(list(zip(self.partition_list, self.model))):

            x_partition = list(torch.split(y, curr_partition, dim=-1))

            for p in range(len(x_partition)):
                for i, layer in reversed(list(enumerate(curr_part_model[p]))):
                    x_partition[p] = layer.backward(x_partition[p])

            y = torch.cat(x_partition, dim=-1)
        y = y.reshape(*input_shape)
        if self.permute_channel:
            y = y.permute(0,3,1,2).contiguous()
        return y, 0.  # out, logdet (0.) not used during sampling

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    def sync(self):  # todo: check the right way of implementing sync
        pass
        # self.weight_inv.copy_(self.weight.data.inverse()) # this is the one in 1x1 conv

    def visualize_weights(self):

        ret = []
        for i, layer in enumerate(self.model[0][0]):
            w1 = layer.w1
            if w1.shape[-2] == 1:
                w1 = w1.expand(-1,-1,layer.l//2, -1)
            level = w1.shape[0]

            ww = torch.zeros(level, 2, 2, layer.l//2, layer.l//2, device=w1.device)
            ww[:,:,:,torch.arange(layer.l//2),torch.arange(layer.l//2)] = w1.transpose(-2,-1)
            www = torch.cat( [torch.cat([ww[:,0,0], ww[:,0,1]], dim=1),torch.cat([ww[:,1,0], ww[:,1,1]], dim=1) ], dim=-1)
            www = www[:1]
            fig_size = 32#min(20 // 6 * www.shape[-1], 32)
            fig, axs = plt.subplots(len(www), figsize=(fig_size, fig_size*len(www)))

            if len(www) == 1:
                axs = [axs]
            for j, d in enumerate(www):
                im = axs[j].imshow(d.detach().cpu().numpy(), cmap='viridis',interpolation='none')
                cbar=fig.colorbar(im,ax=axs[j])
                cbar.ax.tick_params(labelsize=40)

            # fig.savefig(os.path.join(dir, 'weight_level_%03d.png'%i))
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            im = data.reshape((int(h), int(w), -1))
            plt.close()

            ret.append(im)

        return ret

def forward(x, dense_weight):
    x = torch.matmul(x, dense_weight)