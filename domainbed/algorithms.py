# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import numpy as np

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj,
            LARS, shuffle_patches, shuffle_center_patches, replace_with_average_patch,
            voronoi_shuffle, color_histogram_shuffle, phase_scramble, slight_rotate, sobel_edges
    )

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

from domainbed import gsam, sagm

ALGORITHMS = [
    'ERM',
    'ERMPlusPlus',
]

SHUFFLE_PATCH_SIZE=2
SHUFFLE_PATCH_EVERY=100
SHUFFLE_FN="shuffle_center_patches"
GRAD_INHIB_STRENGTH=0.5
DOT_PROD_THRESH=0.2

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


def CGSFactory(BaseAlgoClass, net="network", opt=None):
    class _CGS(BaseAlgoClass, MovingAvg):
        def __init__(self, input_shape, num_classes, num_domains, hparams):
            super().__init__(input_shape, num_classes, num_domains, hparams)

            # --- normalize targets ---
            if isinstance(net, (list, tuple)):
                # multiple modules compose predict(); keep their order
                self._cgs_target_attrs = list(net)
                # make a single container so code sees one .parameters()
                self.network = nn.ModuleList([getattr(self, name) for name in self._cgs_target_attrs])
            else:
                self._cgs_target_attrs = [net]
                # keep update() logic unchanged by ensuring self.network points to the target
                if net != "network":
                    setattr(self, "network", getattr(self, net))

            # --- remember optimizers for CGS steps (only if algo has no self.optimizer) ---
            if isinstance(opt, (list, tuple)):
                self._cgs_opt_attrs = list(opt)
            elif isinstance(opt, str):
                self._cgs_opt_attrs = [opt]
            else:
                self._cgs_opt_attrs = None

            self.grad_inhib_strength = GRAD_INHIB_STRENGTH
            self.shuffle_patch_size  = SHUFFLE_PATCH_SIZE
            self.shuffle_patch_every = SHUFFLE_PATCH_EVERY
            self.dot_prod_thresh     = DOT_PROD_THRESH
            self.shuffle_fn = shuffle_center_patches

            MovingAvg.__init__(self, self.network)

            self.update_step = 0
            self.ret_loss = float("inf")

        def _cgs_zero_grad(self):
            if hasattr(self, "optimizer"):
                self.optimizer.zero_grad()
            elif self._cgs_opt_attrs:
                for name in self._cgs_opt_attrs:
                    getattr(self, name).zero_grad()
            else:
                raise RuntimeError("CGS: no optimizer found; pass opt=... to CGSFactory for this algorithm.")

        def _cgs_step(self):
            if hasattr(self, "optimizer"):
                self.optimizer.step()
            elif self._cgs_opt_attrs:
                for name in self._cgs_opt_attrs:
                    getattr(self, name).step()
            else:
                raise RuntimeError("CGS: no optimizer found; pass opt=... to CGSFactory for this algorithm.")

        def update(self, minibatches, unlabeled=None):
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])

            if self.update_step % self.shuffle_patch_every == 0 and self.update_step != 0:
                # Train on the live network (not SMA)
                t = self.predict(all_x)

                loss = F.cross_entropy(t, all_y)
                self._cgs_zero_grad()
                loss.backward()

                # store original grad
                orig_grads_list = []
                for p in self.network.parameters():
                    if p.grad is not None:
                        orig_grads_list.append(p.grad.clone())
                    else:
                        orig_grads_list.append(None)
                
                self._cgs_zero_grad()
                all_x_shuff = self.shuffle_fn(all_x, patch_size=self.shuffle_patch_size)
                # Train on the live network for shuffled pass
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
                            # new_grad is effectively zero.
                            p.grad = torch.zeros_like(orig_grad_layer)

                    elif orig_grad_layer is not None:
                        # Shuffled gradient is None for this layer, use original gradient
                        p.grad = orig_grad_layer
                    # Else (orig_grad_layer is None):
                    # p.grad remains as it is (which is shuff_grad_layer, possibly None).
                    # This is correct as this parameter was not involved in the original loss.
                                        
                self._cgs_step()
                self.ret_loss = loss.item()

            else:
                # apply the base algorithm's normal update
                out = super().update(minibatches, unlabeled)
                if isinstance(out, dict) and "loss" in out:
                    self.ret_loss = float(out["loss"])

            self.update_step += 1
            self.update_sma()
            return {"loss": self.ret_loss}

        def predict(self, x):
            # TRAIN: preserve base behavior exactly
            if self.training:
                return super().predict(x)

            # EVAL: temporarily swap the target attr(s) to SMA, call base predict, then restore
            self.network_sma.eval()
            if len(self._cgs_target_attrs) == 1:
                attr = self._cgs_target_attrs[0]
                orig = getattr(self, attr)
                try:
                    setattr(self, attr, self.network_sma)
                    with torch.no_grad():
                        return super().predict(x)
                finally:
                    setattr(self, attr, orig)
            else:
                # multiple composing modules; align by order
                origs = []
                try:
                    for i, attr in enumerate(self._cgs_target_attrs):
                        origs.append(getattr(self, attr))
                        setattr(self, attr, self.network_sma[i])
                    with torch.no_grad():
                        return super().predict(x)
                finally:
                    for attr, orig in zip(self._cgs_target_attrs, origs):
                        setattr(self, attr, orig)

    _CGS.__name__ = f"{BaseAlgoClass.__name__}_CGS"
    return _CGS


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

class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}
    
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

class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update_anneal(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        grads_i_v = []
        i = 0
        for x_i, y_i in zip(x, y):
            grads_i = []
            # 1. Compute grad of domain batch
            loss_i = F.cross_entropy(self.predict(x_i), y_i)

            # with open(f"{self.logdir}/loss_{self.test_env}_{i}.txt", "a") as f:
            #     f.write(f"{str(loss_i.item())}\n")
            # f.close()

            # 2. Flat and add domain grads to list
            grad_i = autograd.grad(loss_i, self.network.parameters())
            for g in grad_i:
                grads_i.append(g.flatten())
            grads_i_v.append(torch.cat(grads_i))
            i += 1

        c = list(itertools.combinations(list(range(len(grads_i_v))), 2))

        sims = []
        for i, j in c:
            sims.append(self.cos(grads_i_v[i], grads_i_v[j]))

        if any(s < 0.5 for s in sims):
            # self.anneal()
            # print(f"Loss before: {F.cross_entropy(self.predict(all_x), all_y).item()}")
            start = 10
            finish = 160
            i = 0
            with torch.no_grad():
                for param in self.network.parameters():
                    if start <= i <= finish:
                        # param.add_(torch.randn(param.size()).cuda() * 0.1)

                        param.add_(torch.cuda.FloatTensor.normal_(0, 1))

                        param.add_(torch.Tensor(np.random.uniform(low=self.neighborhoodSize * -1, high=self.neighborhoodSize,
                                                                  size=param.shape)))
                i += 1
            # params = list(self.network.parameters())

            # eps = [torch.Tensor(np.random.uniform(low=self.neighborhoodSize * -1, high=self.neighborhoodSize,
            #                                       size=p.shape)) for p in params]
            #
            # start = 50
            # finish = 155
            # i = 0
            # with torch.no_grad():
            #     for p, v in zip(self.network.parameters(), eps):
            #         if start <= i <= finish:
            #             p.add_(v)
            #     i += 1

        loss = F.cross_entropy(self.predict(all_x), all_y)
        # print(f"Loss after: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()

        # grad_all = self.get_grads()

        # for j in range(len(grads_i_v)):
        #     sim_j = self.cos(grad_all, grads_i_v[j])
        #     with open(f"{self.logdir}/cosine_{self.test_env}_all_{j}.txt", "a") as f:
        #         f.write(f"{str(sim_j.item())}\n")
        #     f.close()

        self.optimizer.step()


        return {"loss": loss.item()}


    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([xi for xi, yi in minibatches])
        all_y = torch.cat([yi for xi, yi in minibatches])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)
        with open("sam_0.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


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

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)
        
#AbstractDANN_CGS = CGSFactory(AbstractDANN)


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

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

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

class GSAM(Algorithm):
    """
    Surrogate Gap Guided Sharpness-Aware Minimization
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = gsam.LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                            min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = gsam.LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.GSAM_optimizer = gsam.GSAM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                        gsam_alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)


    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.GSAM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.GSAM_optimizer.step()
        self.lr_scheduler.step()
        self.GSAM_optimizer.update_rho_t()


        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class SAGM_DG(Algorithm):
    """
    Sharpness-Aware Gradient Matching for Domain Generalization
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = sagm.LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                            min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = sagm.LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.SAGM_optimizer = sagm.SAGM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                   alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

        self.cos = nn.CosineSimilarity(dim=0)

        self.neighborhoodSize = hparams["neighborhoodSize"]
        self.best_loss = np.inf


    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.SAGM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.SAGM_optimizer.step()
        self.lr_scheduler.step()
        self.SAGM_optimizer.update_rho_t()


        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            # network = resnet50_mixstyle_L234_p0d5_a0d1(postion=["conv2_x", "conv3_x", "conv4_x"])
            network = resnet50_mixstyle_L234_p0d5_a0d1(postion=[])

        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


IRM_CGS = CGSFactory(IRM)
VREx_CGS = CGSFactory(VREx)
GroupDRO_CGS = CGSFactory(GroupDRO)
Mixup_CGS = CGSFactory(Mixup)
MMD_CGS = CGSFactory(MMD)
CORAL_CGS = CGSFactory(CORAL)
MLDG_CGS = CGSFactory(MLDG)
MTL_CGS = CGSFactory(MTL)
SagNet_CGS = CGSFactory(
    SagNet,
    net=("network_f", "network_c"),
    opt=("optimizer_f", "optimizer_c")
)
RSC_CGS = CGSFactory(RSC)
Fish_CGS = CGSFactory(Fish)
SAM_CGS = CGSFactory(SAM)
ANDMask_CGS = CGSFactory(ANDMask)
ARM_CGS = CGSFactory(ARM)
GSAM_CGS = CGSFactory(GSAM)
SAGM_DG_CGS = CGSFactory(SAGM_DG)
Mixstyle_CGS = CGSFactory(Mixstyle)