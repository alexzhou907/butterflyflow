import argparse
import copy
import math
import sys

import matplotlib.pyplot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from datasets import load_datasets
import os

import flows.flows1d as fnn
import io
import json

from file_utils import setup_logging

from flows.ema import EMAHelper

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--config',
    type=str,
    required=True)
parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100000,
    help='number of epochs to train (default: 1000)')

parser.add_argument('--valid_epochs', type=int, default=20, metavar='N', help='number of epochs to validate model (default: 50)')

parser.add_argument('--viz_epochs', type=int, default=500, metavar='N', help='number of epochs to validate model (default: 50)')

parser.add_argument('--ckpt_epochs', type=int, default=100, metavar='N', help='number of epochs to validate model (default: 50)')

parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')

parser.add_argument('--warmup_steps', type=int, default=1, metavar='N', help='number of steps to warm up (default: 50)')
parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')

parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')

parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')

parser.add_argument(
    '--ckpt_dir',
    default='outputs')

parser.add_argument(
    '--recover',
    action='store_true',
    default=False,)


parser.add_argument(
    '--load_path',
    default='')


parser.add_argument(
    '--sample',
    action='store_true',
    default=False,)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


params = json.load(open(args.config, 'r'))


args.ema = params.get('use_ema', False)
args.ema_sep = params.get('ema_sep', False)
args.perm_init_type = params.get('perm_init_type', 'iden')

assert args.perm_init_type in ['iden', 'rot', 'orth']

args.lr = params.get('lr', 0.001)
args.perm_lr_mult = params.get('perm_lr_mult', 1)

args.flow = params['flow'].get('type', 'butter_glow')
args.perm_layers = params['flow'].get('perm_layers', 8)
args.levels = params['flow'].get('levels', 2)
args.num_blocks = params['flow'].get('num_blocks', 4)
args.reverse_perm = params['flow'].get('reverse_perm', False)
args.bi_direction = params['flow'].get('bi_direction', False)

params['flow']['weight_init_types'] = [args.perm_init_type for _ in range(args.perm_layers)] if params.get('use_perm', False) else None
params['flow']['bi_direction'] = args.bi_direction

args.use_scheduler = params.get('use_scheduler', False)
args.perm_scheduler = params.get('perm_scheduler', None)

perm_scheduler_parse = ''.join(str(k)+str(v) for k, v in args.perm_scheduler.items()) if args.perm_scheduler is not None else ''

args.dataset =  params.get('dataset')
data_path = params.get('data_path')

print("Using dataset {}".format(args.dataset))

# patient = '3000393'
# patient = '3000063'
# patient = '3000397'
patient = params.get('patient')

train_dataset, test_dataset = load_datasets(args.dataset, data_path, patient=patient)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)



model_path = os.path.join(args.ckpt_dir, args.exp_name)
model_name = os.path.join(model_path, 'model.pt')
checkpoint_name = os.path.join(model_path, 'checkpoint.tar')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)

logger = setup_logging(model_path)


'''models '''

class FlowModel(nn.Module):
    def __init__(self, args, ema:dict=None):
        super().__init__()
        num_inputs = train_dataset.channel_dims
        feat_dims = train_dataset.feat_dims
        num_hidden = {
            'sinetoy': 16,
            'mimic': 64
        }[args.dataset]

        act = 'tanh' if args.dataset is 'GAS' else 'relu'

        modules = nn.ModuleDict()


        if args.flow == 'glow':

            in_channels = num_inputs
            modules = nn.ModuleList()
            for k in range(args.levels):
                level_modules =  nn.ModuleDict()
                in_channels = in_channels * 2
                for j in range(args.num_blocks):
                    level_modules.update({
                        'bn_flow_%d_%d'%(k,j):fnn.ActNorm(in_channels),
                        'luinvmm_%d_%d'%(k,j):fnn.LUInvertibleMM(in_channels),
                        'coupling_%d_%d'%(k,j):fnn.CouplingLayer(
                            in_channels, num_hidden,
                            s_act='tanh', t_act='relu')
                    })

                if k < args.levels - 1:
                    in_channels = in_channels // 2


                modules.append(level_modules)


        elif args.flow == 'butter_glow':
            # mask = torch.arange(0, num_inputs) % 2
            # mask = mask.to(device).float()

            in_channels = num_inputs

            modules = nn.ModuleList()
            for k in range(args.levels):
                level_modules =  nn.ModuleDict()

                in_channels = in_channels * 2
                feat_dims = feat_dims // 2
                for j in range(args.num_blocks):


                    level_modules.update({
                        'bn_flow_%d_%d'%(k,j):fnn.ActNorm(in_channels),
                        'luinvmm_%d_%d'%(k,j):fnn.LUInvertibleMM(in_channels),
                        'butterfly_%d_%d'%(k, j):fnn.general_butterfly_block_simple(in_channels*feat_dims,
                                                                                    layer_num=len(params['flow']['weight_init_types']), share_weights=True,
                                                       use_bias=False, weight_init_types=params['flow']['weight_init_types'],
                                                       reverse_perm=args.reverse_perm,
                                                                                    bi_direction=args.bi_direction),
                        'coupling_%d_%d'%(k,j):fnn.CouplingLayer(
                            in_channels, num_hidden//4,
                            s_act='tanh', t_act='relu')
                    })

                if k < args.levels - 1:
                    in_channels = in_channels // 2

                    if params['flow']['weight_init_types'] is not None:
                        params['flow']['weight_init_types'] = copy.copy(params['flow']['weight_init_types'])
                        params['flow']['weight_init_types'].pop()

                modules.append(level_modules)

                


        model = fnn.FlowSequentialStacked(args, modules)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        self.target_flow = model
        self.ema_helper = None
        self.ema = ema

    def forward(self , *args, **kwargs):
        return self.target_flow.forward(*args, **kwargs)

    def log_probs(self, x, cond_x):
        return self.target_flow.log_probs(x, cond_x)

    def setup_ema(self):  # call after init

        if self.ema is not None:
            if self.ema['type'] == 'all':
                self.ema_model = self.target_flow
            elif self.ema['type'] == 'perm':

                self.ema_model = self.target_flow.get_butterfly()
            else:
                raise NotImplementedError
            self.ema_model_copy = copy.deepcopy(self.ema_model)
            self.ema_helper = EMAHelper(mu=self.ema['ema_rate'])
            self.ema_helper.register(self.ema_model)


    def update_ema(self):

        if self.ema_helper is not None:
            self.ema_helper.update(self.ema_model)
    def set_perm_type(self, use_ema_output=False):
        if self.ema is not None:
            if not use_ema_output:
                if self.ema['type'] == 'perm':
                    self.target_flow.assign_module(self.ema_model)
                elif self.ema['type'] == 'all':
                    self.target_flow = self.ema_model
                else:
                    raise NotImplementedError
            else:
                self.ema_helper.ema(self.ema_model_copy)
                if self.ema['type'] == 'perm':

                    self.target_flow.assign_module(self.ema_model_copy)
                elif self.ema['type'] == 'all':
                    self.target_flow = self.ema_model_copy
                else:
                    raise NotImplementedError

    def ema_state_dict(self):
        if self.ema_helper is not None:
            return self.ema_helper.state_dict()
        else:
            return None

    def get_parameters(self):
        return self.target_flow.get_parameters()


    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True, ema_helper_states ={}):

        super().load_state_dict(state_dict, strict)

        if self.ema_helper is not None:
            self.ema_helper.load_state_dict(ema_helper_states)



model = FlowModel(args, params['ema']).to(device)

polyak_decay = args.polyak
betas = (0.9, polyak_decay)
eps = 1e-8
amsgrad = args.amsgrad
warmups = args.warmup_steps
step_decay = 0.999995

lmbda = lambda step: step / float(warmups) if step < warmups else step_decay ** (step - warmups)
if args.ema and args.ema_sep:

    model_params, butterfly_params = model.target_flow.get_parameters()
    optimizer_perm = optim.Adam([
        {'params': butterfly_params, 'lr': args.lr*args.perm_lr_mult}
    ], betas=betas, eps=eps, amsgrad=amsgrad)

    if args.perm_scheduler is None:
        scheduler_perm = None
    elif args.perm_scheduler['type'] == 'exp':
        scheduler_perm = optim.lr_scheduler.ExponentialLR(optimizer_perm, gamma=args.perm_scheduler['gamma'])
    elif args.perm_scheduler['type'] == 'step':
        scheduler_perm = optim.lr_scheduler.StepLR(optimizer_perm, step_size=args.perm_scheduler['step_size'],
                                                   gamma=args.perm_scheduler['gamma'])
    elif args.perm_scheduler['type'] == 'cos':

        scheduler_perm = optim.lr_scheduler.CosineAnnealingLR(optimizer_perm, T_max=args.perm_scheduler['T_max'],
                                                              eta_min=args.perm_scheduler['eta_min'])
    elif args.perm_scheduler['type'] == 'lambda':

        lmbda_perm = lambda step: step / float(warmups) if step < warmups else step_decay ** (step - warmups)
        scheduler_perm = optim.lr_scheduler.LambdaLR(optimizer_perm, lmbda_perm)
    else:
        raise NotImplementedError


    optimizer = optim.Adam([
        {'params': model_params, 'lr': args.lr}

    ], betas=betas, eps=eps, amsgrad=amsgrad)


else:

    model_params, butterfly_params = model.get_parameters()

    optimizer = optim.Adam([
        {'params': model_params, 'lr': args.lr}
    ] + ([
        {'params': butterfly_params, 'lr': args.lr * args.perm_lr_mult}
    ]  if butterfly_params else []), betas=betas, eps=eps, amsgrad=amsgrad)

    optimizer_perm = None
    scheduler_perm = None

if args.use_scheduler:
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)
else:
    scheduler = None

if args.ema:
    model.setup_ema()

logger.info(args)
logger.info('# of Parameters: %d' % (sum([param.numel() for param in model.target_flow.parameters()])))
global_step = 0


def train(epoch):
    global global_step, writer
    model.train()


    if args.ema and args.ema_sep:
        train_iter = [ optimizer, optimizer_perm]
        train_loss = [0, 0]
        prior_loss = [0, 0]
        jacob_loss = [0, 0]
        use_perm_ema = [True, False]
    else:
        train_iter = [optimizer]
        train_loss = [0, ]
        prior_loss = [0, ]
        jacob_loss = [0, ]
        use_perm_ema = [False]


    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        else:
            cond_data = None
        data = data.to(device)


        for kk, opti in enumerate(train_iter):

            opti.zero_grad()

            if args.ema:
                model.set_perm_type(use_perm_ema[kk])

            neg_loss, res = model.log_probs(data, cond_data)
            loss = -neg_loss.mean()
            prior_loss[kk] += -res['log_prior'].mean()
            jacob_loss[kk] += -res['log_jacob'].mean()
            train_loss[kk] += loss.item()
            loss.backward()
            opti.step()
            if (len(use_perm_ema) == 1 or use_perm_ema[kk]) and scheduler is not None:
                scheduler.step()

            if args.ema and not use_perm_ema[kk]:
                model.update_ema()

            if scheduler is not None:
                curr_lr = scheduler.get_lr()
            else:
                curr_lr = []

            if (len(use_perm_ema) == 1 or use_perm_ema[kk]):
                pbar.update(data.size(0))
                pbar.set_description('Train, NLL per dim in nats: {:.6f}, Prior per dim: {:.6f}, Jacob per dim: {:.6f}'.format(
                    train_loss[kk] / (batch_idx + 1) / np.prod(data.shape[1:]),
                    prior_loss[kk] / (batch_idx + 1) / np.prod(data.shape[1:]),
                    jacob_loss[kk] / (batch_idx + 1) / np.prod(data.shape[1:]),
                    ))

        
    pbar.close()

    if len(train_iter) == 2 and scheduler_perm is not None:
        scheduler_perm.step()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(torch.tensor(train_loader.dataset[0][None], device=data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    val_loss = 0
    prior_loss = 0
    jacob_loss = 0


    if args.ema:
        model.set_perm_type(use_ema_output=True)

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        else:
            cond_data = None
        data = data.to(device)
        with torch.no_grad():
            neg_val_loss, res = model.log_probs(data, cond_data) # sum up batch loss
            val_loss += -neg_val_loss.sum().item()
            prior_loss += -res['log_prior'].sum().item()
            jacob_loss += -res['log_jacob'].sum().item()
        pbar.update(data.size(0))
        pbar.set_description('Val, NLL per dim in nats: {:.6f}, Prior per dim: {:.6f}, Jacob per dim: {:.6f}'.format(
            val_loss / pbar.n / np.prod(data.shape[1:]),
            prior_loss / pbar.n/ np.prod(data.shape[1:]),
            jacob_loss /pbar.n / np.prod(data.shape[1:]),
        ))


    pbar.close()
    return val_loss / len(loader.dataset) / np.prod(data.shape[1:])


def imgs_from_data(data):
    gens = []
    for instance in data:
        # (T, 2)
        instance_np = instance.detach().cpu().numpy()
        pleth = instance_np[:, 0]
        abp = instance_np[:, 1]

        fig, ax = plt.subplots(2, sharex=True)

        ax[0].plot(pleth)
        ax[0].set_ylabel('PLETH')
        ax[1].plot(abp)
        ax[1].set_ylabel('ABP')

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        gens.append(im)
        plt.close()

    return gens


def recon(model, loader):
    logger.info('recon')
    model.eval()

    if args.ema:
        model.set_perm_type(use_ema_output=True)

    data = next(iter(loader)).to(device)

    with torch.no_grad():
        z, _ = model(data, None)

        recon, _ = model(z, mode='reverse')

        # you can plot this





if args.recover:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)
    checkpoint = torch.load(args.load_path)
    start_epoch = checkpoint['epoch']
    patience = checkpoint['patience']
    best_validation_epoch = checkpoint['best_validation_epoch']
    best_validation_loss = checkpoint['best_validation_loss']

    model.load_state_dict(checkpoint['model'], ema_helper_states=checkpoint['ema_helper'] if args.ema else {})
    optimizer.load_state_dict(checkpoint['optimizer'])  # uncomment
    scheduler.load_state_dict(checkpoint['scheduler'])  # uncomment
    if optimizer_perm is not None:
        optimizer_perm.load_state_dict(checkpoint['optimizer_perm'])

    if scheduler_perm is not None:
        scheduler_perm.load_state_dict(checkpoint['scheduler_perm'])

    del checkpoint


else:
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model
    patience=0
    start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    logger.info('\nEpoch: {}'.format(epoch))

    train(epoch)
    if epoch % args.valid_epochs == 0:
        
        validation_loss = validate(epoch, model, test_loader)

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            # best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience+=1


    if epoch == 1 and scheduler is not None:
       for group in optimizer.param_groups:
           del group['initial_lr']
       scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)


    logger.info(
        'Best validation at epoch {}: Average NLL per dim in nats: {:.4f}'.
        format(best_validation_epoch, best_validation_loss))

    
    if epoch % args.viz_epochs == 0:
        recon(model, test_loader)

    if epoch % args.ckpt_epochs == 0 or patient == 0:
        checkpoint = {'epoch': epoch + 1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'ema_helper': model.ema_state_dict(),
                      'best_validation_epoch': best_validation_epoch,
                      'best_validation_loss': best_validation_loss,
                      'patience': patience}
        if optimizer_perm is not None:
            checkpoint['optimizer_perm'] = optimizer_perm.state_dict()

        if scheduler_perm is not None:
            checkpoint['scheduler_perm'] = scheduler_perm.state_dict()
        torch.save(checkpoint, checkpoint_name)

