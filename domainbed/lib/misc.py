# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import copy
import math
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
from collections import Counter
from itertools import cycle
import torch.nn.functional as F


def distance(h1, h2):
    ''' distance of two networks (h1, h2 are classifiers)'''
    dist = 0.
    for param in h1.state_dict():
        h1_param, h2_param = h1.state_dict()[param], h2.state_dict()[param]
        dist += torch.norm(h1_param - h2_param) ** 2  # use Frobenius norms for matrices
    return torch.sqrt(dist)

def proj(delta, adv_h, h):
    ''' return proj_{B(h, \delta)}(adv_h), Euclidean projection to Euclidean ball'''
    ''' adv_h and h are two classifiers'''
    dist = distance(adv_h, h)
    if dist <= delta:
        return adv_h
    else:
        ratio = delta / dist
        for param_h, param_adv_h in zip(h.parameters(), adv_h.parameters()):
            param_adv_h.data = param_h + ratio * (param_adv_h - param_h)
        # print("distance: ", distance(adv_h, h))
        return adv_h

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class ErmPlusPlusMovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 600
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone())/(1.+self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


############################################################
# A general PyTorch implementation of KDE. Builds on:
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py
############################################################

class Kernel(torch.nn.Module):
    """Base class which defines the interface for all kernels."""

    def __init__(self, bw=None):
        super().__init__()
        self.bw = 0.05 if bw is None else bw

    def _diffs(self, test_Xs, train_Xs):
        """Computes difference between each x in test_Xs with all train_Xs."""
        test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
        train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
        return test_Xs - train_Xs

    def forward(self, test_Xs, train_Xs):
        """Computes p(x) for each x in test_Xs given train_Xs."""

    def sample(self, train_Xs):
        """Generates samples from the kernel distribution."""


class GaussianKernel(Kernel):
    """Implementation of the Gaussian kernel."""

    def forward(self, test_Xs, train_Xs):
        diffs = self._diffs(test_Xs, train_Xs)
        dims = tuple(range(len(diffs.shape))[2:])
        if dims == ():
            x_sq = diffs ** 2
        else:
            x_sq = torch.norm(diffs, p=2, dim=dims) ** 2

        var = self.bw ** 2
        exp = torch.exp(-x_sq / (2 * var))
        coef = 1. / torch.sqrt(2 * np.pi * var)

        return (coef * exp).mean(dim=1)

    def sample(self, train_Xs):
        # device = train_Xs.device
        noise = torch.randn(train_Xs.shape) * self.bw
        return train_Xs + noise

    def cdf(self, test_Xs, train_Xs):
        mus = train_Xs                                                      # kernel centred on each observation
        sigmas = torch.ones(len(mus), device=test_Xs.device) * self.bw      # bandwidth = stddev
        x_ = test_Xs.repeat(len(mus), 1).T                                  # repeat to allow broadcasting below
        return torch.mean(torch.distributions.Normal(mus, sigmas).cdf(x_))


def estimate_bandwidth(x, method="silverman"):
    x_, _ = torch.sort(x)
    n = len(x_)
    sample_std = torch.std(x_, unbiased=True)

    if method == 'silverman':
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        iqr = torch.quantile(x_, 0.75) - torch.quantile(x_, 0.25)
        bandwidth = 0.9 * torch.min(sample_std, iqr / 1.34) * n ** (-0.2)

    elif method.lower() == 'gauss-optimal':
        bandwidth = 1.06 * sample_std * (n ** -0.2)

    else:
        raise ValueError(f"Invalid method selected: {method}.")

    return bandwidth


class KernelDensityEstimator(torch.nn.Module):
    """The KernelDensityEstimator model."""

    def __init__(self, train_Xs, kernel='gaussian', bw_select='Gauss-optimal'):
        """Initializes a new KernelDensityEstimator.
        Args:
            train_Xs: The "training" data to use when estimating probabilities.
            kernel: The kernel to place on each of the train_Xs.
        """
        super().__init__()
        self.train_Xs = train_Xs
        self._n_kernels = len(self.train_Xs)

        if bw_select is not None:
            self.bw = estimate_bandwidth(self.train_Xs, bw_select)
        else:
            self.bw = None

        if kernel.lower() == 'gaussian':
            self.kernel = GaussianKernel(self.bw)
        else:
            raise NotImplementedError(f"'{kernel}' kernel not implemented.")

    @property
    def device(self):
        return self.train_Xs.device

    # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an iterative version instead.
    def forward(self, x):
        return self.kernel(x, self.train_Xs)

    def sample(self, n_samples):
        idxs = np.random.choice(range(self._n_kernels), size=n_samples)
        return self.kernel.sample(self.train_Xs[idxs])

    def cdf(self, x):
        return self.kernel.cdf(x, self.train_Xs)


############################################################
# PyTorch implementation of 1D distributions.
############################################################

EPS = 1e-16


class Distribution1D:
    def __init__(self, dist_function=None):
        """
        :param dist_function: function to instantiate the distribution (self.dist).
        :param parameters: list of parameters in the correct order for dist_function.
        """
        self.dist = None
        self.dist_function = dist_function

    @property
    def parameters(self):
        raise NotImplementedError

    def create_dist(self):
        if self.dist_function is not None:
            return self.dist_function(*self.parameters)
        else:
            raise NotImplementedError("No distribution function was specified during intialization.")

    def estimate_parameters(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        return self.create_dist().log_prob(x)

    def cdf(self, x):
        return self.create_dist().cdf(x)

    def icdf(self, q):
        return self.create_dist().icdf(q)

    def sample(self, n=1):
        if self.dist is None:
            self.dist = self.create_dist()
        n_ = torch.Size([]) if n == 1 else (n,)
        return self.dist.sample(n_)

    def sample_n(self, n=10):
        return self.sample(n)


def continuous_bisect_fun_left(f, v, lo, hi, n_steps=32):
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for _ in range(n_steps):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


class Normal(Distribution1D):
    def __init__(self, location=0, scale=1):
        self.location = location
        self.scale = scale
        super().__init__(torch.distributions.Normal)

    @property
    def parameters(self):
        return [self.location, self.scale]

    def estimate_parameters(self, x):
        mean = sum(x) / len(x)
        var = sum([(x_i - mean) ** 2 for x_i in x]) / (len(x) - 1)
        self.location = mean
        self.scale = torch.sqrt(var + EPS)

    def icdf(self, q):
        if q >= 0:
            return super().icdf(q)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            return self.location + self.scale * math.sqrt(-2 * log_y)


class Nonparametric(Distribution1D):
    def __init__(self, use_kde=True, bw_select='Gauss-optimal'):
        self.use_kde = use_kde
        self.bw_select = bw_select
        self.bw, self.data, self.kde = None, None, None
        super().__init__()

    @property
    def parameters(self):
        return []

    def estimate_parameters(self, x):
        self.data, _ = torch.sort(x)

        if self.use_kde:
            self.kde = KernelDensityEstimator(self.data, bw_select=self.bw_select)
            self.bw = torch.ones(1, device=self.data.device) * self.kde.bw

    def icdf(self, q):
        if not self.use_kde:
            # Empirical or step CDF. Differentiable as torch.quantile uses (linear) interpolation.
            return torch.quantile(self.data, float(q))

        if q >= 0:
            # Find quantile via binary search on the KDE CDF
            lo = torch.distributions.Normal(self.data[0], self.bw[0]).icdf(q)
            hi = torch.distributions.Normal(self.data[-1], self.bw[-1]).icdf(q)
            return continuous_bisect_fun_left(self.kde.cdf, q, lo, hi)

        else:
            # To get q *very* close to 1 without numerical issues, we:
            # 1) Use q < 0 to represent log(y), where q = 1 - y.
            # 2) Use the inverse-normal-cdf approximation here:
            #    https://math.stackexchange.com/questions/2964944/asymptotics-of-inverse-of-normal-cdf
            log_y = q
            v = torch.mean(self.data + self.bw * math.sqrt(-2 * log_y))
            return v

# --------------------------------------------------------
# LARS optimizer, implementation from MoCo v3:
# https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


############################################################
# Supervised Contrastive Loss implementation from:
# https://arxiv.org/abs/2004.11362
############################################################
class SupConLossLambda(torch.nn.Module):
    def __init__(self, lamda: float=0.5, temperature: float=0.07):
        super(SupConLossLambda, self).__init__()
        self.temperature = temperature
        self.lamda = lamda

    def forward(self, features: torch.Tensor, labels: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _ = features.shape
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        # create a lookup table for pairwise dot prods
        pairwise_dot_prods = torch.matmul(normalized_features, normalized_features.T)/self.temperature
        loss = 0
        nans = 0
        for i, (label, domain_label) in enumerate(zip(labels, domain_labels)):

            # take the positive and negative samples wrt in/out domain            
            cond_pos_in_domain = torch.logical_and(labels==label, domain_labels == domain_label) # take all positives
            cond_pos_in_domain[i] = False # exclude itself
            cond_pos_out_domain = torch.logical_and(labels==label, domain_labels != domain_label)
            cond_neg_in_domain = torch.logical_and(labels!=label, domain_labels == domain_label)
            cond_neg_out_domain = torch.logical_and(labels!=label, domain_labels != domain_label)

            pos_feats_in_domain = pairwise_dot_prods[cond_pos_in_domain]
            pos_feats_out_domain = pairwise_dot_prods[cond_pos_out_domain]
            neg_feats_in_domain = pairwise_dot_prods[cond_neg_in_domain]
            neg_feats_out_domain = pairwise_dot_prods[cond_neg_out_domain]


            # calculate nominator and denominator wrt lambda scaling
            scaled_exp_term = torch.cat((self.lamda * torch.exp(pos_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(pos_feats_out_domain[:, i])))
            scaled_denom_const = torch.sum(torch.cat((self.lamda * torch.exp(neg_feats_in_domain[:, i]), (1 - self.lamda) * torch.exp(neg_feats_out_domain[:, i]), scaled_exp_term))) + 1e-5

            # nof positive samples
            num_positives = pos_feats_in_domain.shape[0] + pos_feats_out_domain.shape[0] # total positive samples
            log_fraction = torch.log((scaled_exp_term / scaled_denom_const) + 1e-5) # take log fraction
            loss_i = torch.sum(log_fraction) / num_positives
            if torch.isnan(loss_i):
                nans += 1
                continue
            loss -= loss_i # sum and average over num positives
        return loss/(batch_size-nans+1) # avg over batch


def replace_with_average_patch(x, patch_size=4):
    """
    Replaces all patches in each image of the batch with the average patch
    calculated for that specific image.

    Args:
        x (torch.Tensor): Input batch of images (B, C, H, W).
        patch_size (int): The height and width of the square patches. H and W must be divisible by patch_size.

    Returns:
        torch.Tensor: Batch of images where each patch is replaced by the image's average patch (B, C, H, W).
    """
    if x.shape[2] % patch_size != 0 or x.shape[3] % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size")

    batch_size, channels, height, width = x.shape
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    # 1. Extract patches
    # Reshape to [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
    x_reshaped = x.reshape(batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size)
    # Permute to [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
    # Reshape to [B, C, num_patches, patch_size, patch_size]
    patches = x_permuted.reshape(batch_size, channels, num_patches, patch_size, patch_size)

    # 2. Calculate the average patch per image
    # Mean across the 'num_patches' dimension (dim=2)
    # Keepdim=True to maintain dimensions for broadcasting: [B, C, 1, patch_size, patch_size]
    avg_patch = torch.mean(patches, dim=2, keepdim=True)

    # 3. Tile the average patch
    # Expand the singleton dimension to match num_patches: [B, C, num_patches, patch_size, patch_size]
    avg_patches_tiled = avg_patch.expand(-1, -1, num_patches, -1, -1)

    # 4. Reshape back to image format
    # Reshape to [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    avg_patches_tiled = avg_patches_tiled.reshape(batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
    # Permute back to [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
    # Matches the original x_reshaped structure
    x_avg_reshaped = avg_patches_tiled.permute(0, 1, 2, 4, 3, 5)
    # Reshape to final image dimensions [B, C, H, W]
    x_avg_final = x_avg_reshaped.reshape(batch_size, channels, height, width)

    return x_avg_final

def shuffle_patches(x, patch_size=8, center_crop_ratio=0.6):  
    # Calculate the number of patches per image
    num_patches_h = x.shape[2] // patch_size
    num_patches_w = x.shape[3] // patch_size
    num_patches = num_patches_h * num_patches_w

    # Reshape to [batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size]
    x_reshaped = x.reshape(x.shape[0], x.shape[1], num_patches_h, patch_size, num_patches_w, patch_size)

    # Create a random permutation of patches for each image in the batch
    patch_permutation = torch.randperm(num_patches, device=x.device)
    
    # Rearrange the patches based on the permutation
    # [batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size]
    x_shuffled = x_reshaped.permute(0, 1, 2, 4, 3, 5).reshape(x.shape[0], x.shape[1], num_patches_h * num_patches_w, patch_size, patch_size)[:, :, patch_permutation, :, :]
    # Reshape back to original image shape
    x_shuffled = x_shuffled.reshape(x.shape[0], x.shape[1], num_patches_h, num_patches_w, patch_size, patch_size).permute(0, 1, 2, 4, 3, 5).reshape(x.shape)
    return x_shuffled

def shuffle_center_patches(x, center_crop_ratio=0.6, patch_size=8):
    """
    Shuffles patches only in the center crop region of the image, leaving the outer parts unchanged.
    
    Args:
        x: Input tensor of shape [batch_size, channels, height, width]
        center_crop_ratio: Ratio of the center crop (0-1)
        patch_size: Size of patches to shuffle
    
    Returns:
        Tensor with center region shuffled
    """
    batch_size, channels, height, width = x.shape
    
    # Calculate center crop dimensions (ensuring they're multiples of patch_size)
    crop_h = int((height * center_crop_ratio) // patch_size) * patch_size
    crop_w = int((width * center_crop_ratio) // patch_size) * patch_size
    
    # If crop dimensions are 0, make them at least one patch
    if crop_h == 0:
        crop_h = patch_size
    if crop_w == 0:
        crop_w = patch_size
        
    # Calculate start positions to ensure crop is centered
    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2
    
    # Make a copy of the input to modify
    x_shuffled = x.clone()
    
    # Process each image in batch
    for b in range(batch_size):
        # Extract the center crop
        center_crop = x[b, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Calculate patch grid for the center crop
        crop_patches_h = crop_h // patch_size
        crop_patches_w = crop_w // patch_size
        num_patches = crop_patches_h * crop_patches_w
        
        if num_patches <= 1:  # Not enough patches to shuffle
            continue
            
        # Reshape to patches
        center_crop_reshaped = center_crop.reshape(
            channels, crop_patches_h, patch_size, crop_patches_w, patch_size
        ).permute(0, 1, 3, 2, 4)  # [C, crop_patches_h, crop_patches_w, patch_size, patch_size]
        
        # Flatten patches for shuffling
        patches = center_crop_reshaped.reshape(channels, num_patches, patch_size, patch_size)
        
        # Generate random permutation
        perm = torch.randperm(num_patches, device=x.device)
        
        # Shuffle patches
        shuffled_patches = patches[:, perm, :, :]
        
        # Reshape back
        shuffled_crop = shuffled_patches.reshape(
            channels, crop_patches_h, crop_patches_w, patch_size, patch_size
        ).permute(0, 1, 3, 2, 4).reshape(channels, crop_h, crop_w)
        
        # Place shuffled crop back into the image
        x_shuffled[b, :, start_h:start_h+crop_h, start_w:start_w+crop_w] = shuffled_crop
    
    return x_shuffled


def shuffle_within_bbox(x, patch_size, bounding_boxes):
    """
    Shuffles patches within the specified bounding box for each image in the batch.
    
    Args:
        x: Input tensor of shape [batch_size, channels, height, width]
        patch_size: Size of patches to shuffle
        bounding_boxes: List of [top, bottom, left, right] for each image
    
    Returns:
        Tensor with patches shuffled within each bounding box
    """
    batch_size, channels, height, width = x.shape
    x_shuffled = x.clone()
    
    for b in range(batch_size):
        top, bottom, left, right = bounding_boxes[b]
        region = x[b, :, top:bottom, left:right]
        region_h, region_w = region.shape[1], region.shape[2]
        patches_h = region_h // patch_size
        patches_w = region_w // patch_size
        
        if patches_h < 2 or patches_w < 2:  # Need at least 2 patches to shuffle
            print(f"Skipping image {b} due to insufficient patches.")
            continue
        
        # Adjust region to be a multiple of patch_size
        adj_h = patches_h * patch_size
        adj_w = patches_w * patch_size
        region = region[:, :adj_h, :adj_w]
        
        # Reshape to patches
        patches = region.reshape(channels, patches_h, patch_size, patches_w, patch_size)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(channels, patches_h * patches_w, patch_size, patch_size)
        
        # Shuffle patches
        perm = torch.randperm(patches_h * patches_w, device=x.device)
        shuffled_patches = patches[:, perm, :, :]
        
        # Reshape back
        shuffled_region = shuffled_patches.reshape(channels, patches_h, patches_w, patch_size, patch_size)
        shuffled_region = shuffled_region.permute(0, 1, 3, 2, 4).reshape(channels, adj_h, adj_w)
        
        # Place back into the image
        x_shuffled[b, :, top:top + adj_h, left:left + adj_w] = shuffled_region
    
    return x_shuffled


def phase_scramble(x: torch.Tensor,
                   strength: float = 1.0,
                   same_on_batch: bool = False,
                   preserve_dc: bool = True,
                   eps: float = 1e-8,
                   patch_size: int = 8) -> torch.Tensor:
    """
    Phase-scramble a batch of images.

    Args:
        x (Tensor): Input of shape **[B, C, H, W]**, `dtype` float32/float64,   range arbitrary.
        strength (float, default = 1.0):  
            1.0 = fully random phase;  
            0.0 = no scrambling.  
            For 0 < strength < 1 the new phase is an interpolation between the original and random phase.
        same_on_batch (bool): If True, use the *same* random phase mask for every image in the batch
                              (useful for controlled ablations).
        preserve_dc (bool): Keep the DC component (frequency 0,0) unchanged so mean luminance is conserved.
        eps (float): Numerical stability when normalising the inverse FFT.

    Returns:
        Tensor: Phase-scrambled images, same shape & dtype as `x`.
    """
    if x.ndim != 4:
        raise ValueError("expected x to have shape [B, C, H, W]")

    B, C, H, W = x.shape
    device = x.device
    dtype  = x.dtype

    # 2-D FFT over the last two dims
    F = torch.fft.fftn(x, dim=(-2, -1))                       # [B, C, H, W] complex
    amp   = torch.abs(F)                                      # magnitude
    phase = torch.angle(F)                                    # current phase ∈ (-π, π]

    # Sample random phase in (-π, π]
    if same_on_batch:
        rand_phase = (torch.rand(1, 1, H, W, device=device, dtype=dtype) * 2*math.pi) - math.pi
        rand_phase = rand_phase.expand(B, C, H, W)
    else:
        rand_phase = (torch.rand_like(phase) * 2*math.pi) - math.pi

    # Optional interpolation for partial scrambling
    new_phase = phase + strength * (rand_phase - phase)

    # Preserve the DC component (frequency [0,0]) if requested
    if preserve_dc:
        new_phase[..., 0, 0] = phase[..., 0, 0]

    # Re-assemble the spectrum and invert
    scrambled_spec = amp * torch.exp(1j * new_phase)
    x_scrambled = torch.fft.ifftn(scrambled_spec, dim=(-2, -1)).real          # discard tiny imag part

    # Match original numerical range (helps if further processing expects 0‒1 or 0‒255)
    # Shift-and-scale so global mean & std are preserved
    orig_mean, orig_std = x.mean(dim=(-2,-1), keepdim=True), x.std(dim=(-2,-1), keepdim=True) + eps
    new_mean,  new_std  = x_scrambled.mean(dim=(-2,-1), keepdim=True), x_scrambled.std(dim=(-2,-1), keepdim=True) + eps
    x_scrambled = (x_scrambled - new_mean) / new_std * orig_std + orig_mean

    return x_scrambled.to(dtype)

def color_histogram_shuffle(x: torch.Tensor,
                   patch_size: int = 8) -> torch.Tensor:
    """
    Destroys shape and enhances color by sorting all pixels by their RGB values.

    This function perfectly preserves the image's color histogram while
    completely destroying all spatial information. The result is an image
    that represents the original's color palette, sorted and displayed.

    Args:
        x (torch.Tensor): A batch of images in (B, C, H, W) format.

    Returns:
        torch.Tensor: The batch of images with pixels sorted by color.
    """
    batch_size, channels, height, width = x.shape
    device = x.device

    rearranged_batch = []
    for i in range(batch_size):
        img = x[i]

        # 1. Flatten the image into a list of pixels.
        pixels = img.permute(1, 2, 0).reshape(-1, channels) # Shape: (H*W, C)

        # 2. Sort the pixels. This is a lexicographical sort.
        scaled_pixels = (pixels * 255).long()
        # Key = R * 256^2 + G * 256^1 + B * 256^0
        sort_keys = scaled_pixels[:, 0] * 65536 + scaled_pixels[:, 1] * 256 + scaled_pixels[:, 2]

        # Get the indices that would sort the original pixel list
        sorted_indices = torch.argsort(sort_keys)

        # Apply the sorting to the original float pixels
        sorted_pixels = pixels[sorted_indices]

        # 3. Reshape the sorted pixels back into an image.
        new_img = sorted_pixels.reshape(height, width, channels)

        # Permute back to the original (C, H, W) format
        rearranged_batch.append(new_img.permute(2, 0, 1))

    return torch.stack(rearranged_batch)


def voronoi_shuffle(x: torch.Tensor, num_sites: int = 12,
                   patch_size: int = 8) -> torch.Tensor:
    """
    Applies Voronoi shuffling to a batch of images by filling each
    irregular cell with a random CONTIGUOUS CROP from the original image.
    """

    batch_size, channels, height, width = x.shape
    device = x.device

    shuffled_batch = []
    for i in range(batch_size):
        img = x[i]
        shuffled_img = torch.zeros_like(img)

        # 1. Generate random sites for the Voronoi diagram (same as before)
        sites_y = torch.randint(0, height, (num_sites,), device=device)
        sites_x = torch.randint(0, width, (num_sites,), device=device)
        sites = torch.stack([sites_y, sites_x], dim=1).float()

        # 2. Create the Voronoi cell map (same as before)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixel_coords = torch.stack([yy, xx], dim=-1)
        distances_sq = torch.sum(
            (pixel_coords.unsqueeze(2) - sites.view(1, 1, num_sites, 2)) ** 2,
            dim=3
        )
        cell_map = torch.argmin(distances_sq, dim=2)

        # 3. Create the shuffled image by filling cells with random CONTIGUOUS CROPS
        for cell_idx in range(num_sites):
            # Find the pixels belonging to the current cell
            dest_y_coords, dest_x_coords = torch.where(cell_map == cell_idx)

            if dest_y_coords.numel() == 0:
                continue

            # a. Find the bounding box of the destination cell to know the crop size needed.
            min_y, max_y = dest_y_coords.min(), dest_y_coords.max()
            min_x, max_x = dest_x_coords.min(), dest_x_coords.max()
            cell_h, cell_w = max_y - min_y + 1, max_x - min_x + 1

            # b. Pick a single random top-left corner for the source crop.
            #    Ensure the crop fits within the original image dimensions.
            crop_y = torch.randint(0, height - cell_h + 1, (1,)).item()
            crop_x = torch.randint(0, width - cell_w + 1, (1,)).item()
            
            # c. Map the pixels from the source crop to the destination cell.
            src_y_coords = dest_y_coords - min_y + crop_y
            src_x_coords = dest_x_coords - min_x + crop_x

            # d. Copy the pixels from the source crop to the destination cell.
            shuffled_img[:, dest_y_coords, dest_x_coords] = img[:, src_y_coords, src_x_coords]

        shuffled_batch.append(shuffled_img)

    return torch.stack(shuffled_batch)



def slight_rotate(x: torch.Tensor, max_angle_deg: float = 15.0) -> torch.Tensor:
    """
    Applies a small, random rotation to a batch of images. This is a fully
    differentiable transform that encourages learning shape-based features.

    Args:
        x (torch.Tensor): Input batch of images (B, C, H, W).
        max_angle_deg (float): The maximum rotation angle in degrees. A random
                               angle will be chosen between [-max_angle, +max_angle].

    Returns:
        torch.Tensor: A batch of slightly rotated images.
    """
    device = x.device
    batch_size = x.shape[0]

    # 1. Generate a random rotation angle for each image in the batch
    # Convert degrees to radians for torch's trig functions
    angle_rad = torch.deg2rad(
        torch.empty(batch_size, device=device).uniform_(-max_angle_deg, max_angle_deg)
    )
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # 2. Create the affine transformation matrix (differentiable)
    # This is a 2x3 matrix for each image in the batch
    affine_matrix = torch.zeros(batch_size, 2, 3, device=device)
    affine_matrix[:, 0, 0] = cos_a
    affine_matrix[:, 0, 1] = sin_a
    affine_matrix[:, 1, 0] = -sin_a
    affine_matrix[:, 1, 1] = cos_a

    # 3. Create the sampling grid (differentiable)
    # This tells grid_sample where to sample pixels from the source image
    # to form the output image.
    grid = F.affine_grid(
        affine_matrix,
        x.size(),
        align_corners=False # Recommended setting
    )

    # 4. Apply the transformation (differentiable)
    # 'bilinear' mode ensures smooth interpolation, 'border' mode fills
    # new areas with zeros.
    rotated_x = F.grid_sample(
        x,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    return rotated_x


def sobel_edges(x: torch.Tensor, threshold: float = 0.05, blur_sigma: float = 1.0) -> torch.Tensor:
    """
    Applies a differentiable Sobel filter to a batch of images to create edge maps.
    This serves as a proxy for "shape-preserving" information.

    Args:
        x (torch.Tensor): Input batch of images (B, C, H, W).
        threshold (float): A value to suppress low-magnitude edges (noise).
        blur_sigma (float): Standard deviation for an initial Gaussian blur to reduce noise.

    Returns:
        torch.Tensor: A batch of 3-channel edge maps.
    """
    batch_size, channels, height, width = x.shape
    device = x.device

    # 1. Convert to grayscale (differentiable)
    if channels == 3:
        # Standard RGB to grayscale conversion weights
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(device)
        grayscale_x = (x * gray_weights).sum(dim=1, keepdim=True)
    else:
        grayscale_x = x

    # 2. (Optional but recommended) Apply a slight Gaussian blur to reduce noise (differentiable)
    if blur_sigma > 0:
        # Create a Gaussian kernel
        kernel_size = int(6 * blur_sigma) + 1
        if kernel_size % 2 == 0: kernel_size += 1
        kernel = torch.exp(-(torch.arange(kernel_size, device=device) - kernel_size // 2)**2 / (2 * blur_sigma**2))
        kernel_x = (kernel / kernel.sum()).view(1, 1, 1, kernel_size)
        kernel_y = (kernel / kernel.sum()).view(1, 1, kernel_size, 1)
        # Apply blur horizontally and vertically
        grayscale_x = F.conv2d(F.conv2d(grayscale_x, kernel_x, padding='same'), kernel_y, padding='same')

    # 3. Define Sobel kernels for detecting horizontal and vertical edges
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    # 4. Apply Sobel filters using convolution (differentiable)
    grad_x = F.conv2d(grayscale_x, sobel_x, padding=1)
    grad_y = F.conv2d(grayscale_x, sobel_y, padding=1)

    # 5. Calculate gradient magnitude (differentiable)
    # Use torch.hypot for numerical stability: sqrt(a^2 + b^2)
    edge_magnitude = torch.hypot(grad_x, grad_y)

    # 6. Normalize the edge map per image to a [0, 1] range
    # This prevents images with naturally higher contrast from dominating
    norm_edge_map = torch.zeros_like(edge_magnitude)
    for i in range(batch_size):
        img_mag = edge_magnitude[i]
        min_val, max_val = img_mag.min(), img_mag.max()
        if max_val > min_val:
            norm_edge_map[i] = (img_mag - min_val) / (max_val - min_val)

    # 7. Apply a soft threshold (differentiable clamp) or a hard one
    # A simple hard threshold is non-differentiable but often works in practice
    # as we only care about the gradient *where there are edges*.
    if threshold > 0:
        norm_edge_map[norm_edge_map < threshold] = 0.

    # 8. Replicate the single-channel edge map to 3 channels to match network input
    # The network must see a tensor of the same shape as the original.
    final_output = norm_edge_map.repeat(1, channels, 1, 1)

    return final_output