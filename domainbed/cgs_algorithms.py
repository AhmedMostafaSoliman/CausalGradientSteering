# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

import math
import copy
import numpy as np
import inspect
from collections import OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
            LARS,  SupConLossLambda, shuffle_patches, shuffle_center_patches, replace_with_average_patch,
            voronoi_shuffle, color_histogram_shuffle, phase_scramble, slight_rotate, sobel_edges
    )

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
    
    

class ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule(Algorithm):
    """
    same as ERM_GRAD_INHIB_FAST_conditional, but with normalized 2 (a different normalization approach) gradients PER LAYER
        new_grad = orig_grad - factor * shuff_grad
        scale = orig_grad.norm() / (new_grad.norm() + 1e-8)
        param.grad = scale * new_grad
    
    note here the dot product threshold is bigger than 0.3 not just 0
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_GRAD_INHIB_FAST_dp_thresh_layerSchedule, self).__init__(input_shape, num_classes, num_domains, hparams)
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
    


class ERM_GRAD_INHIB_FAST_dp_thresh_layerScheduleSGD(Algorithm):
    """
    same as ERM_GRAD_INHIB_FAST_conditional, but with normalized 2 (a different normalization approach) gradients PER LAYER
        new_grad = orig_grad - factor * shuff_grad
        scale = orig_grad.norm() / (new_grad.norm() + 1e-8)
        param.grad = scale * new_grad
    
    note here the dot product threshold is bigger than 0.3 not just 0
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_GRAD_INHIB_FAST_dp_thresh_layerScheduleSGD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            momentum=0)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    self.optimizer, T_max=self.hparams["linear_steps"]
        #)
        self.grad_inhib_strength = hparams["grad_inhib_strength"]
        # Allow runtime updates from a scheduler by keeping this mutable
        self.shuffle_patch_size = hparams["shuffle_patch_size"]
        self.shuffle_patch_every = hparams["shuffle_patch_every"]
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
                        # new_grad is effectively zero; set grad to zeros of same shape
                        p.grad = torch.zeros_like(orig_grad_layer)

                elif orig_grad_layer is not None:
                    # Shuffled gradient is None for this layer, use original gradient
                    p.grad = orig_grad_layer
                # Else (orig_grad_layer is None):
                # p.grad remains as it is (which is shuff_grad_layer, possibly None).
                # This is correct as this parameter was not involved in the original loss.
                                    
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
        #self.scheduler.step()
        return {"loss": self.ret_loss}
        
    def predict(self, x):
        return self.network(x)

    # Optional helper for external schedulers
    def set_shuffle_patch_size(self, new_size: int):
        self.shuffle_patch_size = int(new_size)
    def set_center_crop_ratio(self, new_ratio: float):
        self.center_crop_ratio = float(new_ratio)
    def set_grad_inhib_strength(self, new_strength: float):
        self.grad_inhib_strength = float(new_strength)
    


class ERM_GRAD_INHIB_FAST_dp_thresh_layerScheduleAdamW(Algorithm):
    """
    same as ERM_GRAD_INHIB_FAST_conditional, but with normalized 2 (a different normalization approach) gradients PER LAYER
        new_grad = orig_grad - factor * shuff_grad
        scale = orig_grad.norm() / (new_grad.norm() + 1e-8)
        param.grad = scale * new_grad
    
    note here the dot product threshold is bigger than 0.3 not just 0
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_GRAD_INHIB_FAST_dp_thresh_layerScheduleAdamW, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.grad_inhib_strength = hparams["grad_inhib_strength"]
        # Allow runtime updates from a scheduler by keeping this mutable
        self.shuffle_patch_size = hparams["shuffle_patch_size"]
        self.shuffle_patch_every = hparams["shuffle_patch_every"]
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
                        # new_grad is effectively zero; set grad to zeros of same shape
                        p.grad = torch.zeros_like(orig_grad_layer)

                elif orig_grad_layer is not None:
                    # Shuffled gradient is None for this layer, use original gradient
                    p.grad = orig_grad_layer
                # Else (orig_grad_layer is None):
                # p.grad remains as it is (which is shuff_grad_layer, possibly None).
                # This is correct as this parameter was not involved in the original loss.
                                    
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
    def set_center_crop_ratio(self, new_ratio: float):
        self.center_crop_ratio = float(new_ratio)
    def set_grad_inhib_strength(self, new_strength: float):
        self.grad_inhib_strength = float(new_strength)
    
