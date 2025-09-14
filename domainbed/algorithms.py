# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
import inspect
import math
import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os
import random

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
            LARS,  SupConLossLambda, shuffle_patches, shuffle_center_patches, replace_with_average_patch,
            voronoi_shuffle, color_histogram_shuffle, phase_scramble, slight_rotate, sobel_edges
    )
import contextlib

ALGORITHMS = [
    'ERM',
    'ERMPlusPlus',
    'ERM_GRAD_INHIB_FAST_SAL',
    'ERM_GRAD_INHIB_ENTROPY',
    'ERM_GRAD_INHIB_FAST_conditional_normalized_layer_with_viz',
    'ERM_GRAD_INHIB_FAST_conditional_normalized_layer',
    'ERM_GRAD_INHIB_FAST_conditional_normalized',
    'ERM_GRAD_INHIB_FAST_conditional'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
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



class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERM_GRAD_INHIB_FAST_conditional_normalized2_dp_thresh_layer(Algorithm, MovingAvg):
    """
    same as ERM_GRAD_INHIB_FAST_conditional, but with normalized 2 (a different normalization approach) gradients PER LAYER
        new_grad = orig_grad - factor * shuff_grad
        scale = orig_grad.norm() / (new_grad.norm() + 1e-8)
        param.grad = scale * new_grad
    
    note here the dot product threshold is bigger than 0.3 not just 0
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        # SMA components
        MovingAvg.__init__(self, self.network)
        self.grad_inhib_strength = hparams["grad_inhib_strength"]
        self.shuffle_patch_size = hparams["shuffle_patch_size"]
        self.shuffle_patch_every = hparams["shuffle_patch_every"]
        self.shuffle_fn = hparams["shuffle_fn"]
        self.dot_prod_thresh = hparams.get("dot_prod_thresh", 0.3)
        # convert the shuffle_fn string to a function
        if self.shuffle_fn == 'shuffle_patches':
            self.shuffle_fn = shuffle_patches
        elif self.shuffle_fn == 'shuffle_center_patches':
            self.shuffle_fn = shuffle_center_patches
        elif self.shuffle_fn == 'replace_with_average_patch':
            self.shuffle_fn = replace_with_average_patch
        else:
            self.shuffle_fn = globals()[hparams["shuffle_fn"]]

        self.update_step=0
        import math
        self.ret_loss = math.inf

    def update(self, minibatches, unlabeled=None):        
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        if self.update_step % self.shuffle_patch_every == 0 and self.update_step != 0:
            # Train on the live network (not SMA)
            t = self.network(all_x)

            loss = F.cross_entropy(t, all_y)
            self.optimizer.zero_grad()
            loss.backward()

            # store original grad
            orig_grads_list = []
            for p in self.network.parameters():
                if p.grad is not None:
                    orig_grads_list.append(p.grad.clone())
                else:
                    orig_grads_list.append(None)
            
            self.optimizer.zero_grad()
            all_x_shuff = self.shuffle_fn(all_x, patch_size=self.shuffle_patch_size)
            # Train on the live network for shuffled pass
            t_sh = self.network(all_x_shuff)
            loss_shuff = F.cross_entropy(t_sh, all_y)
            loss_shuff.backward()
            
            # Store shuffled gradients
            shuff_grads_list = []
            for p in self.network.parameters():
                if p.grad is not None:
                    shuff_grads_list.append(p.grad.clone())
                else:
                    shuff_grads_list.append(None)

            # Process gradients layer by layer
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad_layer = orig_grads_list[p_idx]
                # p.grad currently holds the gradient from loss_shuff.backward()
                # which is shuff_grads_list[p_idx]
                shuff_grad_layer = p.grad 

                if orig_grad_layer is not None and shuff_grad_layer is not None:
                    orig_flat_layer = orig_grad_layer.flatten()
                    shuff_flat_layer = shuff_grad_layer.flatten()

                    dot_prod_layer = torch.dot(orig_flat_layer, shuff_flat_layer)
                    shuff_norm_sq_layer = torch.sum(shuff_flat_layer * shuff_flat_layer)
                    
                    projection_scalar_layer = dot_prod_layer / (shuff_norm_sq_layer + 1e-8)

                    if dot_prod_layer > self.dot_prod_thresh:
                        factor_layer = self.grad_inhib_strength * projection_scalar_layer
                    else:
                        factor_layer = 0
                    
                    # Calculate new gradient for the layer
                    new_grad_layer = orig_grad_layer - factor_layer * shuff_grad_layer
                    
                    # Layer-specific normalization
                    orig_norm_layer = orig_grad_layer.norm()
                    new_grad_norm_val_layer = new_grad_layer.norm()
                    
                    # Avoid division by zero if new_grad_norm_val_layer is close to zero
                    if new_grad_norm_val_layer > 1e-8:
                        scale_layer = orig_norm_layer / new_grad_norm_val_layer
                        p.grad = scale_layer * new_grad_layer
                    else:
                        # new_grad is effectively zero.
                        p.grad = torch.zeros_like(orig_grad_layer)

                elif orig_grad_layer is not None:
                    # Shuffled gradient is None for this layer, use original gradient
                    p.grad = orig_grad_layer
                # Else (orig_grad_layer is None):
                # p.grad remains as it is (which is shuff_grad_layer, possibly None).
                # This is correct as this parameter was not involved in the original loss.
                                    
            self.optimizer.step()
            self.ret_loss = loss.item()
        else:
            # Standard supervised step on live network
            t = self.network(all_x)

            loss = F.cross_entropy(t, all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.ret_loss = loss.item()
     
        self.update_step += 1
        # Update SMA after updating live weights
        self.update_sma()
        return {"loss": self.ret_loss}
        
    def predict(self, x):
        # Use the averaged model for evaluation
        self.network_sma.eval()
        return self.network_sma(x)


class ERMPlusPlus(Algorithm,ErmPlusPlusMovingAvg):
    """
    Empirical Risk Minimization with improvements (ERM++)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self,input_shape, num_classes, num_domains,hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        if self.hparams["lars"]:
            self.optimizer = LARS(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        linear_parameters = []
        for n, p in self.network[1].named_parameters():
            linear_parameters.append(p)

        if self.hparams["lars"]:
            self.linear_optimizer = LARS(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )

        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=self.hparams["linear_lr"],
                weight_decay=self.hparams['weight_decay'],
                foreach=False
            )
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience = 1)
        ErmPlusPlusMovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):

        if self.global_iter > self.hparams["linear_steps"]:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer



        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)

        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        self.update_sma()
        if not self.hparams["freeze_bn"]:
            self.network_sma.train()
            self.network_sma(all_x)

        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)

    def set_lr(self, eval_loaders_iid=None, schedule=None,device=None):
        with torch.no_grad():
             if self.global_iter > self.hparams["linear_steps"]:
                 if schedule is None:
                     self.network_sma.eval()
                     val_losses = []
                     for loader in eval_loaders_iid:
                         loss = 0.0
                         for x, y in loader:
                             x = x.to(device)
                             y = y.to(device)
                             loss += F.cross_entropy(self.network_sma(x),y)
                         val_losses.append(loss / len(loader ))
                     val_loss = torch.mean(torch.stack(val_losses))
                     self.scheduler.step(val_loss)
                     self.lr_schedule.append(self.scheduler._last_lr)
                     if len(self.lr_schedule) > 1:
                         if self.lr_schedule[-1] !=  self.lr_schedule[-2]:
                            self.lr_schedule_changes += 1
                     if self.lr_schedule_changes == 3:
                         self.lr_schedule[-1] = [0.0]
                     return self.lr_schedule
                 else:
                     self.optimizer.param_groups[0]['lr'] = (torch.Tensor(schedule[0]).requires_grad_(False))[0]
                     schedule = schedule[1:]
             return schedule



class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}



class IRM_CGS(IRM):
    """
    Invariant Risk Minimization (IRM) with Conditional Gradient Shaping (CGS).

    CGS is applied as a gradient regularizer on top of the standard IRM update loop.
    It modifies the gradient from the IRM loss before the optimizer step.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        # The parent __init__ sets up self.network, self.optimizer, self.update_count, etc.
        super(IRM_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        # CGS-specific hyperparameters and state
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4

        # Use a separate counter for CGS to not interfere with IRM's annealing logic
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        # Check if it's time for a CGS-enhanced update
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])

            # 1. Calculate original gradients from the full IRM loss
            self.optimizer.zero_grad()

            penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                              >= self.hparams['irm_penalty_anneal_iters'] else 1.0)
            nll = 0.
            penalty = 0.
            all_logits = self.network(all_x)
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                nll += F.cross_entropy(logits, y)
                penalty += self._irm_penalty(logits, y)
            nll /= len(minibatches)
            penalty /= len(minibatches)
            orig_loss = nll + (penalty_weight * penalty)
            orig_loss.backward()

            # Store original gradients from the IRM loss
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate "shortcut" gradients from a simple cross-entropy loss on shuffled data
            all_x_shuff = self.shuffle_fn(
                all_x,
                center_crop_ratio=self.center_crop_ratio,
                patch_size=self.shuffle_patch_size,
            )
            t_sh = self.predict(all_x_shuff) # self.predict is self.network()
            loss_shuff = F.cross_entropy(t_sh, all_y)
            loss_shuff.backward()

            # 3. Apply CGS gradient modification logic
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad_layer = orig_grads_list[p_idx]
                shuff_grad_layer = p.grad # p.grad currently holds the gradient from loss_shuff

                if orig_grad_layer is not None and shuff_grad_layer is not None:
                    orig_flat_layer = orig_grad_layer.flatten()
                    shuff_flat_layer = shuff_grad_layer.flatten()

                    dot_prod_layer = torch.dot(orig_flat_layer, shuff_flat_layer)
                    shuff_norm_sq_layer = torch.sum(shuff_flat_layer * shuff_flat_layer)
                    projection_scalar_layer = dot_prod_layer / (shuff_norm_sq_layer + 1e-8)

                    if dot_prod_layer > self.dot_prod_thresh:
                        factor_layer = self.grad_inhib_strength * projection_scalar_layer
                    else:
                        factor_layer = 0

                    new_grad_layer = orig_grad_layer - factor_layer * shuff_grad_layer
                    orig_norm_layer = orig_grad_layer.norm()
                    new_grad_norm_val_layer = new_grad_layer.norm()

                    if new_grad_norm_val_layer > 1e-8:
                        scale_layer = orig_norm_layer / new_grad_norm_val_layer
                        p.grad = scale_layer * new_grad_layer
                    else:
                        p.grad = torch.zeros_like(orig_grad_layer)
                elif orig_grad_layer is not None:
                    p.grad = orig_grad_layer
                # Else: p.grad is already what it should be (None or from shuff_loss, which is fine to ignore)

            # 4. Handle optimizer reset (logic from parent IRM)
            if self.update_count.item() == self.hparams['irm_penalty_anneal_iters']:
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay'])

            # 5. Step the optimizer and update counters
            self.optimizer.step()
            self.update_count += 1
            self.cgs_update_step += 1
            return {'loss': orig_loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}
        else:
            # --- Standard IRM Update Step ---
            # For non-CGS steps, just call the parent's update method.
            # This ensures all original logic (like resetting Adam, updating update_count) is executed correctly.
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result




class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}



class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)



# =====================================================================================
# V-REx with Conditional Gradient Shaping (CGS)
# =====================================================================================
class VREx_CGS(VREx):
    """
    V-REx with Conditional Gradient Shaping (CGS).

    CGS is applied as a gradient regularizer on top of the standard V-REx update loop.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0
        # CGS-specific hyperparameters and state

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])

            # 1. Calculate original gradients from the full V-REx loss
            self.optimizer.zero_grad()
            penalty_weight = self.hparams["vrex_lambda"] if self.update_count >= self.hparams["vrex_penalty_anneal_iters"] else 1.0
            all_logits = self.network(all_x)
            all_logits_idx = 0
            losses = torch.zeros(len(minibatches))
            for i, (x, y) in enumerate(minibatches):
                logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                losses[i] = F.cross_entropy(logits, y)
            mean = losses.mean()
            penalty = ((losses - mean) ** 2).mean()
            orig_loss = mean + penalty_weight * penalty
            orig_loss.backward()

            # Store original gradients
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate "shortcut" gradients from shuffled data
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS gradient modification logic (as in IRM_CGS)
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad
            
            # 4. Handle optimizer reset (logic from parent V-REx)
            if self.update_count.item() == self.hparams['vrex_penalty_anneal_iters']:
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay']
                )

            self.optimizer.step()
            self.update_count += 1
            self.cgs_update_step += 1
            return {'loss': orig_loss.item(), 'nll': mean.item(), 'penalty': penalty.item()}
        else:
            # --- Standard V-REx Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result

# =====================================================================================
# GroupDRO with Conditional Gradient Shaping (CGS)
# =====================================================================================
class GroupDRO_CGS(GroupDRO):
    """
    GroupDRO with Conditional Gradient Shaping (CGS).

    CGS is applied as a gradient regularizer on top of the standard GroupDRO update loop.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            device = "cuda" if minibatches[0][0].is_cuda else "cpu"
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            
            # 1. Calculate original gradients from the full GroupDRO loss
            self.optimizer.zero_grad()
            if not len(self.q):
                self.q = torch.ones(len(minibatches)).to(device)
            losses = torch.zeros(len(minibatches)).to(device)
            for m in range(len(minibatches)):
                x, y = minibatches[m]
                losses[m] = F.cross_entropy(self.predict(x), y)
                self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()
            self.q /= self.q.sum()
            orig_loss = torch.dot(losses, self.q)
            orig_loss.backward()

            # Store original gradients
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate "shortcut" gradients from shuffled data
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS gradient modification logic
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad

            self.optimizer.step()
            self.cgs_update_step += 1
            return {'loss': orig_loss.item()}
        else:
            # --- Standard GroupDRO Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result
            
# =====================================================================================
# AbstractMMD (base for CORAL and MMD) with Conditional Gradient Shaping (CGS)
# =====================================================================================
class AbstractMMD_CGS(AbstractMMD):
    """
    Abstract MMD class with Conditional Gradient Shaping (CGS).
    This serves as a base for CORAL_CGS and MMD_CGS.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD_CGS, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian)
        
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])

            # 1. Calculate original gradients from the full MMD/CORAL loss
            self.optimizer.zero_grad()
            objective = 0
            penalty = 0
            nmb = len(minibatches)
            features = [self.featurizer(xi) for xi, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [yi for _, yi in minibatches]
            for i in range(nmb):
                objective += F.cross_entropy(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    penalty += self.mmd(features[i], features[j])
            objective /= nmb
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)
            orig_loss = objective + (self.hparams['mmd_gamma'] * penalty)
            orig_loss.backward()

            # Store original gradients
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate "shortcut" gradients from shuffled data
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS gradient modification logic
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad

            self.optimizer.step()
            self.cgs_update_step += 1
            penalty_item = penalty.item() if torch.is_tensor(penalty) else penalty
            return {'loss': objective.item(), 'penalty': penalty_item}
        else:
            # --- Standard MMD/CORAL Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result

class MMD_CGS(AbstractMMD_CGS):
    """ MMD using Gaussian kernel with CGS """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD_CGS, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)

class CORAL_CGS(AbstractMMD_CGS):
    """ MMD using mean and covariance difference with CGS """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL_CGS, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)





class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


# Add this import at the top of your algorithms.py file if it's not already there
# =====================================================================================
# Abstract DANN with Conditional Gradient Shaping (CGS)
# =====================================================================================
class AbstractDANN_CGS(AbstractDANN):
    """
    Abstract base class for DANN and CDANN with Conditional Gradient Shaping (CGS).

    CGS is integrated into the generator's update step to regularize the featurizer.
    The discriminator's update remains unchanged.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):
        super(AbstractDANN_CGS, self).__init__(
            input_shape, num_classes, num_domains, hparams, conditional, class_balance
        )

        # CGS-specific hyperparameters
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1

        # Common data preparation for all steps
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        disc_input = all_z + self.class_embeddings(all_y) if self.conditional else all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        # Calculate discriminator loss (needed for both disc and gen steps)
        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        # Decide which component to update
        d_steps_per_g = self.hparams['d_steps_per_g_step']
        is_disc_step = self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g
        is_cgs_step = not is_disc_step and self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0

        if is_disc_step:
            # --- Original Discriminator Update Step ---
            input_grad = autograd.grad(
                F.cross_entropy(disc_out, disc_labels, reduction='sum'),
                [disc_input], create_graph=True)[0]
            grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
            disc_loss_with_penalty = disc_loss + self.hparams['grad_penalty'] * grad_penalty

            self.disc_opt.zero_grad()
            disc_loss_with_penalty.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss_with_penalty.item()}

        elif is_cgs_step:
            # --- CGS-Enhanced Generator Update Step ---
            self.cgs_update_step += 1

            # 1. Calculate original gradients from the full generator loss
            self.gen_opt.zero_grad()
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            orig_gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
            orig_gen_loss.backward()

            # Store original gradients for generator's parameters
            gen_params = list(self.featurizer.parameters()) + list(self.classifier.parameters())
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in gen_params]
            self.gen_opt.zero_grad()

            # 2. Calculate "shortcut" gradients from a simple classification loss
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS gradient modification logic
            for p_idx, p in enumerate(gen_params):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad
            
            self.disc_opt.zero_grad()
            self.gen_opt.step()
            return {'gen_loss': orig_gen_loss.item()}

        else:
            # --- Original Generator Update Step ---
            self.cgs_update_step += 1
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
            
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

# =====================================================================================
# Final DANN and CDANN Classes with CGS
# =====================================================================================

class DANN_CGS(AbstractDANN_CGS):
    """Unconditional DANN with Conditional Gradient Shaping"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN_CGS, self).__init__(
            input_shape, num_classes, num_domains, hparams,
            conditional=False, class_balance=False
        )

class CDANN_CGS(AbstractDANN_CGS):
    """Conditional DANN with Conditional Gradient Shaping"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN_CGS, self).__init__(
            input_shape, num_classes, num_domains, hparams,
            conditional=True, class_balance=True
        )




class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}




# =====================================================================================
# MLDG with Conditional Gradient Shaping (CGS)
# =====================================================================================
class MLDG_CGS(MLDG):
    """
    Model-Agnostic Meta-Learning (MLDG) with Conditional Gradient Shaping (CGS).

    CGS is applied to the final accumulated gradient from the MLDG procedure.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0
    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            num_mb = len(minibatches)
            objective = 0
            
            # 1. Manually accumulate the original gradients from the MLDG process
            self.optimizer.zero_grad()
            for p in self.network.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
                inner_net = copy.deepcopy(self.network)
                inner_opt = torch.optim.Adam(
                    inner_net.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay']
                )
                inner_obj = F.cross_entropy(inner_net(xi), yi)
                inner_opt.zero_grad()
                inner_obj.backward()
                inner_opt.step()
                objective += inner_obj.item()
                for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                    if p_src.grad is not None:
                        p_tgt.grad.data.add_(p_src.grad.data / num_mb)

                loss_inner_j = F.cross_entropy(inner_net(xj), yj)
                grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)
                objective += (self.hparams['mldg_beta'] * loss_inner_j).item()
                for p, g_j in zip(self.network.parameters(), grad_inner_j):
                    if g_j is not None:
                        p.grad.data.add_(self.hparams['mldg_beta'] * g_j.data / num_mb)
            
            # Store the final accumulated MLDG gradients
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate shortcut gradients
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS logic
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad
            
            self.optimizer.step()
            self.cgs_update_step += 1
            return {'loss': objective / len(minibatches)}
        else:
            # --- Standard MLDG Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result

# =====================================================================================
# MTL with Conditional Gradient Shaping (CGS)
# =====================================================================================
# =====================================================================================
# MTL with Conditional Gradient Shaping (CGS) - CORRECTED
# =====================================================================================
class MTL_CGS(MTL):
    """
    Marginal Transfer Learning (MTL) with Conditional Gradient Shaping (CGS).
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.grad_inhib_strength = hparams.get("grad_inhib_strength", 1.0)
        self.shuffle_patch_size = hparams["shuffle_patch_size"]
        self.shuffle_patch_every = hparams.get("shuffle_patch_every", 2)
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = hparams.get("center_crop_ratio", 0.6)
        self.dot_prod_thresh = hparams.get("dot_prod_thresh", 0.3)
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            
            # 1. Calculate original gradients from the MTL loss
            self.optimizer.zero_grad()
            orig_loss = 0
            for env, (x, y) in enumerate(minibatches):
                orig_loss += F.cross_entropy(self.predict(x, env), y)
            orig_loss.backward()

            # Store original gradients
            # !!!!! CORRECTED LINE: Get parameters from featurizer and classifier separately !!!!!
            all_params = list(self.featurizer.parameters()) + list(self.classifier.parameters())
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in all_params]
            self.optimizer.zero_grad()

            # 2. Calculate shortcut gradients
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            
            # For shortcut loss on shuffled data, use a domain-neutral prediction
            features = self.featurizer(all_x_shuff)
            avg_embedding = self.embeddings.mean(0).view(1, -1).repeat(len(features), 1)
            shuff_logits = self.classifier(torch.cat((features, avg_embedding), 1))
            loss_shuff = F.cross_entropy(shuff_logits, all_y)
            loss_shuff.backward()
            
            # 3. Apply CGS logic
            for p_idx, p in enumerate(all_params):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad

            self.optimizer.step()
            self.cgs_update_step += 1
            return {'loss': orig_loss.item()}
        else:
            # --- Standard MTL Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result


# =====================================================================================
# SagNet with Conditional Gradient Shaping (CGS)
# =====================================================================================
class SagNet_CGS(SagNet):
    """
    Style Agnostic Network (SagNet) with Conditional Gradient Shaping (CGS).
    CGS is applied only to the "content" learning step.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        is_cgs_step = self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0
        
        if is_cgs_step:
            # --- CGS-Enhanced Content Learning Step ---
            self.optimizer_f.zero_grad()
            self.optimizer_c.zero_grad()
            
            # 1. Calculate original gradients from the content loss
            loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
            loss_c.backward()
            
            content_params = list(self.network_f.parameters()) + list(self.network_c.parameters())
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in content_params]
            
            self.optimizer_f.zero_grad()
            self.optimizer_c.zero_grad()
            
            # 2. Calculate shortcut gradients
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            # Use the standard prediction path for shortcut loss
            shuff_logits = self.network_c(self.network_f(all_x_shuff))
            loss_shuff = F.cross_entropy(shuff_logits, all_y)
            loss_shuff.backward()

            # 3. Apply CGS logic
            for p_idx, p in enumerate(content_params):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad

            self.optimizer_f.step()
            self.optimizer_c.step()

        else: # Standard SagNet content update
            self.optimizer_f.zero_grad()
            self.optimizer_c.zero_grad()
            loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
            loss_c.backward()
            self.optimizer_f.step()
            self.optimizer_c.step()

        # --- Style and Adversarial steps are unchanged ---
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()
        
        self.cgs_update_step += 1
        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

# =====================================================================================
# RSC with Conditional Gradient Shaping (CGS)
# =====================================================================================
class RSC_CGS(RSC):
    """
    Representation Self-Challenging (RSC) with Conditional Gradient Shaping (CGS).
    CGS is applied to the final gradient computed by the RSC procedure.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for _, y in minibatches])

            # 1. Calculate original gradients from the full RSC loss
            self.optimizer.zero_grad()
            # This logic is copied from the parent RSC class
            all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
            all_f = self.featurizer(all_x)
            all_p = self.classifier(all_f)
            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]
            percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
            percentiles = torch.Tensor(percentiles).unsqueeze(1).repeat(1, all_g.size(1))
            mask_f = all_g.lt(percentiles.to(all_g.device)).float()
            all_f_muted = all_f * mask_f
            all_p_muted = self.classifier(all_f_muted)
            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), self.drop_b)
            mask_b = changes.lt(percentile).float().view(-1, 1)
            mask = torch.logical_or(mask_f, mask_b).float()
            all_p_muted_again = self.classifier(all_f * mask)
            orig_loss = F.cross_entropy(all_p_muted_again, all_y)
            orig_loss.backward()

            # Store original gradients
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in self.network.parameters()]
            self.optimizer.zero_grad()

            # 2. Calculate shortcut gradients
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS logic
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad
            
            self.optimizer.step()
            self.cgs_update_step += 1
            return {'loss': orig_loss.item()}
        else:
            # --- Standard RSC Update Step ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result




class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)



# =====================================================================================
# ARM with Conditional Gradient Shaping (CGS)
# =====================================================================================
class ARM_CGS(ARM):
    """
    Adaptive Risk Minimization (ARM) with Conditional Gradient Shaping (CGS).
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ARM_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        self.grad_inhib_strength = 0.6
        self.shuffle_patch_size = 1
        self.shuffle_patch_every = 200 
        self.shuffle_fn = shuffle_center_patches
        self.center_crop_ratio = 1.0
        self.dot_prod_thresh = 0.4
        self.cgs_update_step = 0

    def update(self, minibatches, unlabeled=None):
        if self.cgs_update_step > 0 and self.cgs_update_step % self.shuffle_patch_every == 0:
            # --- CGS Update Step ---
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            
            # 1. Calculate original gradients from the ARM loss
            self.optimizer.zero_grad()
            orig_loss = F.cross_entropy(self.predict(all_x), all_y)
            orig_loss.backward()

            # Store original gradients for the whole network (featurizer, classifier, and context_net)
            all_params = list(self.network.parameters()) + list(self.context_net.parameters())
            orig_grads_list = [p.grad.clone() if p.grad is not None else None for p in all_params]
            self.optimizer.zero_grad()

            # 2. Calculate shortcut gradients
            all_x_shuff = self.shuffle_fn(
                all_x, center_crop_ratio=self.center_crop_ratio, patch_size=self.shuffle_patch_size
            )
            # ARM's predict method modifies the input, so it must be used for both losses.
            loss_shuff = F.cross_entropy(self.predict(all_x_shuff), all_y)
            loss_shuff.backward()

            # 3. Apply CGS logic
            for p_idx, p in enumerate(all_params):
                orig_grad, shuff_grad = orig_grads_list[p_idx], p.grad
                if orig_grad is not None and shuff_grad is not None:
                    dot_prod = torch.dot(orig_grad.flatten(), shuff_grad.flatten())
                    if dot_prod > self.dot_prod_thresh:
                        proj_scalar = dot_prod / (torch.sum(shuff_grad * shuff_grad) + 1e-8)
                        factor = self.grad_inhib_strength * proj_scalar
                        new_grad = orig_grad - factor * shuff_grad
                        new_norm = new_grad.norm()
                        if new_norm > 1e-8:
                            p.grad = (orig_grad.norm() / new_norm) * new_grad
                        else:
                            p.grad = torch.zeros_like(orig_grad)
                    else:
                        p.grad = orig_grad
                elif orig_grad is not None:
                    p.grad = orig_grad

            self.optimizer.step()
            self.cgs_update_step += 1
            return {'loss': orig_loss.item()}
        else:
            # --- Standard ARM Update Step (which is just ERM's update) ---
            result = super().update(minibatches, unlabeled)
            self.cgs_update_step += 1
            return result




class ERM_CGS(Algorithm):
    """
    same as ERM_GRAD_INHIB_FAST_conditional, but with normalized 2 (a different normalization approach) gradients PER LAYER
        new_grad = orig_grad - factor * shuff_grad
        scale = orig_grad.norm() / (new_grad.norm() + 1e-8)
        param.grad = scale * new_grad
    
    note here the dot product threshold is bigger than 0.3 not just 0
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_CGS, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.grad_inhib_strength = hparams["grad_inhib_strength"]
        # Allow runtime updates from a scheduler by keeping this mutable
        self.shuffle_patch_size = hparams["shuffle_patch_size"]
        self.shuffle_patch_every = hparams["shuffle_patch_every"]
        # Always use center shuffle for simplicity
        self.shuffle_fn = shuffle_center_patches
        # Center crop ratio (can be scheduled)
        self.center_crop_ratio = hparams.get("center_crop_ratio_start", hparams.get("center_crop_ratio", 0.6))
        self.dot_prod_thresh = hparams.get("dot_prod_thresh", 0.3)


        self.update_step=0
        import math
        self.ret_loss = math.inf

    def update(self, minibatches, unlabeled=None):        
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        if self.update_step % self.shuffle_patch_every == 0 and self.update_step != 0:
            t = self.predict(all_x)

            loss = F.cross_entropy(t, all_y)
            self.optimizer.zero_grad()
            loss.backward()

            # store original grad
            orig_grads_list = []
            for p in self.network.parameters():
                if p.grad is not None:
                    orig_grads_list.append(p.grad.clone())
                else:
                    orig_grads_list.append(None)
            
            self.optimizer.zero_grad()
            # Use the (possibly scheduled) patch size and center crop ratio
            all_x_shuff = self.shuffle_fn(
                all_x,
                center_crop_ratio=self.center_crop_ratio,
                patch_size=self.shuffle_patch_size,
            )
            t_sh = self.predict(all_x_shuff)
            loss_shuff = F.cross_entropy(t_sh, all_y)
            loss_shuff.backward()
            
            # Store shuffled gradients
            shuff_grads_list = []
            for p in self.network.parameters():
                if p.grad is not None:
                    shuff_grads_list.append(p.grad.clone())
                else:
                    shuff_grads_list.append(None)

            # Process gradients layer by layer
            for p_idx, p in enumerate(self.network.parameters()):
                orig_grad_layer = orig_grads_list[p_idx]
                # p.grad currently holds the gradient from loss_shuff.backward()
                # which is shuff_grads_list[p_idx]
                shuff_grad_layer = p.grad 

                if orig_grad_layer is not None and shuff_grad_layer is not None:
                    orig_flat_layer = orig_grad_layer.flatten()
                    shuff_flat_layer = shuff_grad_layer.flatten()

                    dot_prod_layer = torch.dot(orig_flat_layer, shuff_flat_layer)
                    shuff_norm_sq_layer = torch.sum(shuff_flat_layer * shuff_flat_layer)
                    
                    projection_scalar_layer = dot_prod_layer / (shuff_norm_sq_layer + 1e-8)

                    if dot_prod_layer > self.dot_prod_thresh:
                        factor_layer = self.grad_inhib_strength * projection_scalar_layer
                    else:
                        factor_layer = 0
                    
                    # Calculate new gradient for the layer
                    new_grad_layer = orig_grad_layer - factor_layer * shuff_grad_layer
                    
                    # Layer-specific normalization
                    orig_norm_layer = orig_grad_layer.norm()
                    new_grad_norm_val_layer = new_grad_layer.norm()
                    
                    # Avoid division by zero if new_grad_norm_val_layer is close to zero
                    if new_grad_norm_val_layer > 1e-8:
                        scale_layer = orig_norm_layer / new_grad_norm_val_layer
                        p.grad = scale_layer * new_grad_layer
                    else:
                        # new_grad is effectively zero
                        p.grad = torch.zeros_like(orig_grad_layer)

                elif orig_grad_layer is not None:
                    # Shuffled gradient is None for this layer, use original gradient
                    p.grad = orig_grad_layer
                                    
            self.optimizer.step()
            # Return the supervised loss value for logging
            self.ret_loss = loss.item()
        else:
            t = self.predict(all_x)

            loss = F.cross_entropy(t, all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.ret_loss = loss.item()
     
        self.update_step += 1
        return {"loss": self.ret_loss}
        
    def predict(self, x):
        return self.network(x)

    # Optional helper for external schedulers
    def set_shuffle_patch_size(self, new_size: int):
        self.shuffle_patch_size = int(new_size)
    # Optional helpers for external schedulers
    def set_center_crop_ratio(self, new_ratio: float):
        self.center_crop_ratio = float(new_ratio)
    def set_grad_inhib_strength(self, new_strength: float):
        self.grad_inhib_strength = float(new_strength)
    