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
