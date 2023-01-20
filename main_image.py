import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import argparse
import random
import math
import numpy as np
import pathlib
import copy

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_


from datasets import load_datasets, get_batch, preprocess, postprocess, permute, get_permute_all, get_permute_matrix, logit_transform, sigmoid_transform
from flows import FlowGenModel, VDeQuantFlowGenModel, EMAHelper
from file_utils import setup_logging

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')
parser.add_argument('--batch-steps', type=int, default=1, metavar='N', help='number of steps for each batch (the batch size of each step is batch-size / steps (default: 1)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N', help='number of epochs to train')
parser.add_argument('--warmup_steps', type=int, default=10, metavar='N', help='number of steps to warm up (default: 50)')
parser.add_argument('--valid_epochs', type=int, default=10, metavar='N', help='number of epochs to validate model (default: 50)')
parser.add_argument('--valid_steps', type=int, default=100000, metavar='N', help='number of epochs to validate model (default: 50)')
parser.add_argument('--ckpt_epochs', type=int, default=5, metavar='N', help='number of epochs to validate model (default: 50)')
parser.add_argument('--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=6700417, metavar='S', help='random seed (default: 6700417)')
parser.add_argument('--train_k', type=int, default=1, metavar='N', help='training K (default: 1)')
parser.add_argument('--n_bits', type=int, default=8, metavar='N', help='number of bits per pixel.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--polyak', type=float, default=0.999, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
parser.add_argument('--dequant', choices=['uniform', 'variational'], help='dequantization method', default='uniform')
parser.add_argument('--ckpt_dir', help='path for saving model file.', default='outputs')
parser.add_argument('--recover', action='store_true', help='recover the model from disk.')
parser.add_argument('--load_path', help='load path.', default='')
parser.add_argument('--no_data_permute',action='store_true',  default=False)
parser.add_argument('--sample',action='store_true',  default=False)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

torch.backends.cudnn.benchmark = True


def get_permutation(init, ds=None):
    
    b, c, h, w = init.shape
    data_dim = c * h * w
    fn = './random_perm{}.pth'.format('_'+ds if ds is not None else '')
    if pathlib.Path(fn).exists():

        perm_all =  torch.load(fn)#[torch.randperm(data_dim)]
    else:
        perm_all = [torch.randperm(data_dim)]
        torch.save(perm_all, fn)
    model_layers = 8

    return perm_all, model_layers

def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = [torch.empty_like(p_) for p_ in p]
    for k in range(len(s)):
        s[k][p[k]] = torch.arange(len(p[k]))
    return s
# save_image(init[:16], os.path.join(result_path, 'init_permute.png'), nrow=4)


# breakpoint()
def get_optimizer(learning_rate, parameters):
    if opt == 'adam':
        return optim.Adam(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)
    elif opt == 'adamax':
        return optim.Adamax(parameters, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError('unknown optimization method: %s' % opt)

def total_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train(epoch, k, step):
    logger.info('Epoch: %d (lr=%.6f (%s), patient=%d)' % (epoch, lr, opt, patient))
    fgen.train()

    start_time = time.time()

    if args.ema and args.ema_sep:
        train_iter = [ optimizer, optimizer_perm]
        nll = [0, 0]
        nent = [0, 0]
        num_insts = [0, 0]
        num_back = [0, 0]
        num_nans = [0, 0]
        use_perm_ema = [True, False]
    else:
        train_iter = [optimizer]
        nll = [0, ]
        nent = [0, ]
        num_insts = [0, ]
        num_back = [0, ]
        num_nans = [0, ]
        use_perm_ema = [False]

    for batch_idx, (data, _) in enumerate(train_loader):
        step += 1
        batch_size = len(data)
        data = data.to(device, non_blocking=True)
        if not args.no_data_permute:
            data = permute(data, perm_all)
        nll_batch = [0 for _ in range(len(nll))]
        nent_batch = [0 for _ in range(len(nll))]
        data_list = [data, ] if batch_steps == 1 else data.chunk(batch_steps, dim=0)

        for kk, opti in enumerate(train_iter):

            opti.zero_grad()

            if args.ema:
                fgen.set_perm_type(use_perm_ema[kk])

            for data in data_list:

                x = preprocess(data, n_bits)

                # [batch, k]
                noise, log_probs_posterior = fgen.dequantize(x, nsamples=k)

                # [batch, k] -> [1]
                log_probs_posterior = log_probs_posterior.mean(dim=1).sum()

                # [batch, k, channels, H, W] -> [batch, channels, H, W]
                data = preprocess(data, n_bits, noise[:, 0:1]).squeeze(1)

                if args.use_logits:
                    data = logit_transform(data)
                # print(nx)
                ## train perm layer
                log_probs = fgen.log_probability(data).sum()

                loss = (log_probs_posterior - log_probs) / batch_size

                loss.backward()

                with torch.no_grad():
                    nll_batch[kk] -= log_probs.item()
                    nent_batch[kk] += log_probs_posterior.item()


            if grad_clip > 0:
                grad_norm = clip_grad_norm_(fgen.parameters(), grad_clip)
            else:
                grad_norm = total_grad_norm(fgen.parameters())

            if math.isnan(grad_norm):
                num_nans[kk] += 1
            else:

                opti.step()
                if (len(use_perm_ema) == 1 or use_perm_ema[kk]):
                    scheduler.step()


                num_insts[kk] += batch_size
                nll[kk] += nll_batch[kk]
                nent[kk] += nent_batch[kk]

                if args.ema and not use_perm_ema[kk]:
                    fgen.update_ema()


            train_nent = nent_batch[kk] / batch_size
            train_nll = nll_batch[kk] / batch_size + train_nent + np.log(n_bins/2 ) * nx
            bits_per_pixel = train_nll / (nx * np.log(2.0))
            nent_per_pixel = train_nent / (nx * np.log(2.0))
            curr_lr = scheduler.get_lr()

            if batch_idx % args.log_interval == 0 and (len(use_perm_ema) == 1 or use_perm_ema[kk]):
                sys.stdout.write("\b" * num_back[kk])
                sys.stdout.write(" " * num_back[kk])
                sys.stdout.write("\b" * num_back[kk])

                log_info = '[{}/{} ({:.0f}%) lr={}, {}] NLL: {:.2f}, BPD: {:.4f}, NENT: {:.2f}, NEPD: {:.4f}'.format(
                    batch_idx * batch_size, len(train_index), 100. * batch_idx * batch_size / len(train_index), curr_lr, num_nans[kk],
                    train_nll, bits_per_pixel, train_nent, nent_per_pixel)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back[kk] = len(log_info)


    if len(train_iter) == 2:
        scheduler_perm.step()

    train_nent_all, train_nll_all, bits_per_pixel_all, nent_per_pixel_all = [], [] ,[] ,[]
    for kk in range(len(nll)):
        sys.stdout.write("\b" * num_back[kk])
        sys.stdout.write(" " * num_back[kk])
        sys.stdout.write("\b" * num_back[kk])
        train_nent = nent[kk] / num_insts[kk]
        train_nll = nll[kk] / num_insts[kk] + train_nent + np.log(n_bins / 2.) * nx
        bits_per_pixel = train_nll / (nx * np.log(2.0))
        nent_per_pixel = train_nent / (nx * np.log(2.0))
        logger.info('TRAIN {} Average NLL: {:.2f}, BPD: {:.4f}, NENT: {:.2f}, NEPD: {:.4f}, time: {:.1f}s'.format(
            '(ema)' if use_perm_ema[kk] else '',
            train_nll, bits_per_pixel, train_nent, nent_per_pixel, time.time() - start_time))
            

        train_nent_all.append(train_nent)
        train_nll_all.append(train_nll)
        bits_per_pixel_all.append(bits_per_pixel)
        nent_per_pixel_all.append(nent_per_pixel)

    return train_nll_all, bits_per_pixel_all, train_nent_all, nent_per_pixel_all, step

def eval(epoch, data_loader, k, step, init=False): # add init flag to initialize butterfly block parameters

    fgen.eval()

    nent = 0
    nll_mc = 0
    nll_iw = 0
    num_insts = 0
    nepd = 0


    if not init and args.ema:
        fgen.set_perm_type(use_ema_output=True)


    for i, (data, _) in enumerate(data_loader):
        data = data.to(device, non_blocking=True)
        if not args.no_data_permute:
            data = permute(data, perm_all)
        # [batch, channels, H, W]
        batch, c, h, w = data.size()
        x = preprocess(data, n_bits)
        # [batch, k]
        noise, log_probs_posterior = fgen.dequantize(x, nsamples=k)
        # [batch, k, channels, H, W] -> [batch, channels, H, W]
        data = preprocess(data, n_bits, noise)

        # [batch * k, channels, H, W] -> [batch * k] -> [batch, k]
        log_probs = fgen.log_probability(data.view(batch * k, c, h, w)).view(batch, k)
        # [batch, k]
        log_iw = log_probs - log_probs_posterior

        num_insts += batch
        nent += log_probs_posterior.mean(dim=1).sum().item()
        nll_mc -= log_iw.mean(dim=1).sum().item()
        nll_iw += (math.log(k) - torch.logsumexp(log_iw, dim=1)).sum().item()
        if init: # added
            return # added

    nent = nent / num_insts
    nepd = nent / (nx * np.log(2.0))
    nll_mc = nll_mc / num_insts + np.log(n_bins / 2.) * nx
    bpd_mc = nll_mc / (nx * np.log(2.0))
    nll_iw = nll_iw / num_insts + np.log(n_bins / 2.) * nx
    bpd_iw = nll_iw / (nx * np.log(2.0))

    logger.info('TEST {}: Avg  NLL: {:.2f}, NENT: {:.2f}, IW: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}, BPD_IW: {:.4f}'.format(
        '(ema)' if args.ema else '',
        nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw))


    return nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw


def sample(epoch):
    logger.info('sampling')
    fgen.eval()

    if args.ema:
        fgen.set_perm_type(use_ema_output=True)
    n = 64 #256 # edited: smaller batch size
    taus = [0.7, 0.8, 0.9, 1.0]
    for t in taus:
        z = torch.randn(n, nc, imageSize, imageSize).to(device)
        z = z * t
        img, _ = fgen.decode(z)

        if args.use_logits:
            img = sigmoid_transform(img)
        img = postprocess(img, n_bits)

        if not args.no_data_permute:
            img =permute(img, invert_permutation(perm_all))

        image_file = 'sample{}.t{:.1f}.png'.format(epoch, t)
        save_image(img, os.path.join(result_path, image_file), nrow=8)



def visualize_weights():
    logger.info('weight_visualizing')
    fgen.eval()

    if args.ema:
        fgen.set_perm_type(use_ema_output=True)

    imgs = fgen.target_flow.visualize_weights()

    # save_image(img, os.path.join(result_path, image_file), nrow=8)
    for i, blocks in enumerate(imgs):
        for j, steps in enumerate(blocks):
            # breakpoint()
            for l, level in enumerate(steps):
                plt.imsave(os.path.join(result_path, f'weights_block{i}_steps{j}_level{l}.png'), level)
            wandb.log({f'weights_block{i}_steps{j}': [wandb.Image(im) for im in steps]})


params = json.load(open(args.config, 'r'))

imageSize = int(params.get('image_size',32))
nc = int(params['flow']['in_channels'])
nx = imageSize * imageSize * nc
n_bits = args.n_bits
n_bins = 2. ** n_bits
test_k = 8
train_k = args.train_k

args.ema = params.get('use_ema', False)
args.ema_sep = params.get('ema_sep', False)
args.perm_type = params.get('perm_type', 1)
args.perm_init_type = params.get('perm_init_type', 'iden')

assert args.perm_init_type in ['iden', 'rot', 'orth']

args.lr = params.get('lr', 0.001)
args.perm_lr_mult = params.get('perm_lr_mult', 1)

args.use_conv1by1 = params['flow'].get('use_conv1by1', False)
args.perm_layers = params['flow'].get('perm_layers', 8)
args.use_intermediate_perm = params['flow'].get('use_intermediate_perm', False)
args.reverse_perm = params['flow'].get('reverse_perm', False)
args.factors = params['flow'].get('factors', [2 for _ in range(params['flow']['levels'])])
args.LU_decomposed =  params['flow'].get('LU_decomposed', False)
args.bi_direction = params['flow'].get('bi_direction', False)
args.permute_channel = params['flow'].get('permute_channel', False)
args.hidden_channels = params['flow'].get('hidden_channels', None)

args.perm_scheduler = params.get('perm_scheduler', {'type': 'lambda'})

perm_scheduler_parse = ''.join(str(k)+str(v) for k, v in args.perm_scheduler.items())
''' data '''

dataset =  params.get('dataset')
data_path = params.get('data_path')
args.use_logits = params.get('use_logits', False)
train_data, test_data = load_datasets(dataset, data_path)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
test_index = np.arange(len(test_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True)  # edit: added drop_last=True
test_loader = DataLoader(test_data, batch_size=500, shuffle=False, num_workers=args.workers, pin_memory=True,
                         drop_last=True)  # edit: added drop_last=True
batch_steps = args.batch_steps

if not args.no_data_permute:
    init = next(iter(train_loader))[0].to(device)
    perm_all, model_layers = get_permutation( init=init, ds=dataset)


flow_name = params['flow']['type']
model_path = os.path.join(args.ckpt_dir, args.exp_name)
model_name = os.path.join(model_path, 'model.pt')
checkpoint_name = os.path.join(model_path, 'checkpoint_epoch{}.tar')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)

logger = setup_logging(model_path)

logger.info(len(train_index))
logger.info(len(test_index))

logger.info(args)
polyak_decay = args.polyak
opt = args.opt
betas = (0.9, polyak_decay)
eps = 1e-8
amsgrad = params.pop('amsgrad', False)
warmups = args.warmup_steps
step_decay = params.pop('step_decay', 0.999997)
grad_clip = args.grad_clip
dequant = args.dequant


# pop unused ones to prevent invalid input for the model

params.pop('use_ema', None)
params.pop('ema_sep', None)
params.pop('perm_type', None)
params.pop('perm_init_type', None)
params.pop('lr', None)
params.pop('perm_lr_mult', None)
params['flow'].pop('perm_layers', None)
params.pop('perm_scheduler', None)
params.pop('image_size',None)
params.pop('use_logits', None)
params.pop('dataset', None)
params.pop('data_path')

params['flow']['weight_init_types'] = [args.perm_init_type for _ in range(args.perm_layers)] if params.pop('use_perm', False) else None
params['flow']['factors'] = args.factors
params['flow']['LU_decomposed'] = args.LU_decomposed
params['flow']['bi_direction'] = args.bi_direction
params['flow']['permute_channel'] = args.permute_channel
params['flow']['hidden_channels'] = args.hidden_channels

params['recover'] = args.recover
if 'ema' in params:
    assert not (params['ema']['type'] == 'all' and args.ema_sep)


''' model '''
if dequant == 'uniform':
    fgen = FlowGenModel.from_params(params).to(device)
elif dequant == 'variational':
    fgen = VDeQuantFlowGenModel.from_params(params).to(device)
else:
    raise ValueError('unknown dequantization method: %s' % dequant)
# initialize
init_batch_size = 2048
if flow_name == 'nsf':
    # avoid OOM error
    init_batch_size = 128
init_index = np.random.choice(train_index, init_batch_size, replace=False)
init_data, _ = get_batch(train_data, init_index)
init_data = preprocess(init_data.to(device), n_bits)

if args.use_logits:
    init_data = sigmoid_transform(init_data)
#breakpoint()
if not args.no_data_permute:
    init_data = permute(init_data, perm_all)

fgen.eval()

fgen.init(init_data, init_scale=1.0)



fgen.to_device(device)

lmbda = lambda step: step / float(warmups) if step < warmups else step_decay ** (step - warmups)
if args.ema and args.ema_sep:

    model_params, butterfly_params = fgen.target_flow.get_parameters()
    optimizer_perm = optim.Adam([
        {'params': butterfly_params, 'lr': args.lr*args.perm_lr_mult}
    ], betas=betas, eps=eps, amsgrad=amsgrad)

    if args.perm_scheduler['type'] == 'exp':
        scheduler_perm = optim.lr_scheduler.ExponentialLR(optimizer_perm, gamma=args.perm_scheduler['gamma'])
    elif args.perm_scheduler['type'] == 'step':
        scheduler_perm = optim.lr_scheduler.StepLR(optimizer_perm, step_size=args.perm_scheduler['step_size'],
                                                   gamma=args.perm_scheduler['gamma'])
    elif args.perm_scheduler['type'] == 'multistep':
        scheduler_perm = optim.lr_scheduler.MultiStepLR(optimizer_perm, milestones=[250, 350, 450], gamma=0.1) #original
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

    model_params, butterfly_params = fgen.target_flow.get_parameters()

    optimizer = optim.Adam([
        {'params': model_params, 'lr': args.lr}
    ] + ([
        {'params': butterfly_params, 'lr': args.lr * args.perm_lr_mult}
    ]  if butterfly_params else []), betas=betas, eps=eps, amsgrad=amsgrad)

    optimizer_perm = None
    scheduler_perm = None

if args.ema:
    fgen.setup_ema()

if args.recover:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)
    checkpoint = torch.load(args.load_path, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    step = checkpoint['step']
    patient = checkpoint['patient']
    best_epoch = checkpoint['best_epoch']
    best_nll_mc = checkpoint['best_nll_mc']
    best_bpd_mc = checkpoint['best_bpd_mc']
    best_nll_iw = checkpoint['best_nll_iw']
    best_bpd_iw = checkpoint['best_bpd_iw']
    best_nent = checkpoint['best_nent']
    best_nepd = checkpoint['best_nepd']

    with torch.no_grad():
        eval(0, test_loader, test_k, 0, init=True) # added to initialize the butterfly flow model structure, otherwise the keys cannot be loaded

    fgen.load_state_dict(checkpoint['model'], ema_helper_states=checkpoint['ema_helper'] if args.ema else None)

    optimizer.load_state_dict(checkpoint['optimizer']) # uncomment
    scheduler.load_state_dict(checkpoint['scheduler']) # uncomment
    if optimizer_perm is not None:
        optimizer_perm.load_state_dict(checkpoint['optimizer_perm'])

    if scheduler_perm is not None:
        scheduler_perm.load_state_dict(checkpoint['scheduler_perm'])

    del checkpoint
    


else:
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)

    start_epoch = 1
    step = 0
    patient = 0
    best_epoch = 0
    best_nll_mc = 1e12
    best_bpd_mc = 1e12
    best_nll_iw = 1e12
    best_bpd_iw = 1e12
    best_nent = 1e12
    best_nepd = 1e12



# number of parameters
logger.info('# of Parameters: %d' % (sum([param.numel() for param in fgen.target_flow.parameters()])))
fgen.sync()
lr_min = args.lr / 100
lr = scheduler.get_lr()[0]
checkpoint_epochs = args.ckpt_epochs


for epoch in range(start_epoch, args.epochs + 1):
    train_nll, bits_per_pixel, train_nent, nent_per_pixel, step = train(epoch, train_k, step)
    fgen.sync()
    logger.info('-' * 100)
    
    if epoch ==1 or epoch ==5 or (epoch < 5000 and epoch % 10 == 0) or epoch % args.valid_epochs == 0:
    
        with torch.no_grad():
            nll_mc, nent, nll_iw, bpd_mc, nepd, bpd_iw = eval(epoch, test_loader, test_k, step)

        # bpd_curve_test.append((epoch, bpd_mc))
        if nll_mc < best_nll_mc:
            patient = 0
            torch.save(fgen.state_dict(), model_name)

            best_epoch = epoch
            best_nll_mc = nll_mc
            best_nll_iw = nll_iw
            best_bpd_mc = bpd_mc
            best_bpd_iw = bpd_iw
            best_nent = nent
            best_nepd = nepd

        else:
            patient += 1

    logger.info('Best NLL: {:.2f}, NENT: {:.2f}, IW: {:.2f}, BPD: {:.4f}, NEPD: {:.4f}, BPD_IW: {:.4f}, epoch: {}'.format(
        best_nll_mc, best_nent, best_nll_iw, best_bpd_mc, best_nepd, best_bpd_iw, best_epoch))
    logger.info('=' * 100)

    if epoch == 1:
       for group in optimizer.param_groups:
           del group['initial_lr']
       scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)

    lr = scheduler.get_lr()[0]

    if epoch % checkpoint_epochs == 0:
    
        checkpoint = {'epoch': epoch + 1,
                      'step': step+1,
                      'model': fgen.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'ema_helper': fgen.ema_state_dict(),
                      'best_epoch': best_epoch,
                      'best_nll_mc': best_nll_mc,
                      'best_bpd_mc': best_bpd_mc,
                      'best_nll_iw': best_nll_iw,
                      'best_bpd_iw': best_bpd_iw,
                      'best_nent': best_nent,
                      'best_nepd': best_nepd,
                      'patient': patient}
        if optimizer_perm is not None:
            checkpoint['optimizer_perm'] = optimizer_perm.state_dict()

        if scheduler_perm is not None:
            checkpoint['scheduler_perm'] = scheduler_perm.state_dict()
        torch.save(checkpoint, os.path.join(model_path, 'checkpoint.tar'))


