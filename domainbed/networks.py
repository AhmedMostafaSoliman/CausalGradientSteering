# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

import timm

from domainbed.models.lora.lora_util import inject_trainable_moe_kronecker_new, inject_trainable_moe_kronecker_inhib

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
    
"""
#from dinov2.models.vision_transformer import vit_small_inhib, vit_small, vit_base_inhib
#from dinov2.hub.utils import _DINOV2_BASE_URL, _make_dinov2_model_name

class DinoV2(torch.nn.Module):
    """ """
    def __init__(self,input_shape, hparams):
        super(DinoV2, self).__init__()

        self.network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        #print network no. of paramters
        print("Number of parameters in DinoV2: ", sum(p.numel() for p in self.network.parameters()))
        self.n_outputs =  5 * 768

        nc = input_shape[0]

        if nc != 3:
            raise RuntimeError("Inputs must have 3 channels")

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['vit_dropout'])

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'attn' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def forward(self, x):
        x = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([
            x[0][1],
            x[1][1],
            x[2][1],
            x[3][1],
            x[3][0].mean(1)
            ], dim=1)
        return self.dropout(linear_input)




class CLIPInhib(nn.Module):
    def get_module(self, module, name):
        for n, m in module.named_modules():
            if n == name:
                return m

    def build_blocks(self, model, block_name_dict):
        #  blocks = nn.ModuleList()
        blocks = []  # saved model can be broken...
        for _key, name_list in block_name_dict.items():
            block = nn.ModuleList()
            for module_name in name_list:
                module = self.get_module(model, module_name)
                block.append(module)
            blocks.append(block)

        return blocks
    
    def freeze_(self, model):
        #Freeze model
        #Note that this function does not control BN
        
        for p in model.parameters():
            p.requires_grad_(False)
    
    def hook(self, module, input, output):
        self._features.append(output)

    def build_feature_hooks(self, feats, block_names):
        assert feats in ["stem_block", "block"]

        if feats is None:
            return []

        # build feat layers
        if feats.startswith("stem"):
            last_stem_name = block_names["stem"][-1]
            feat_layers = [last_stem_name]
        else:
            feat_layers = []

        for name, module_names in block_names.items():
            if name == "stem":
                continue

            module_name = module_names[-1]
            feat_layers.append(module_name)

        #  print(f"feat layers = {feat_layers}")

        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)

        return feat_layers

    def __init__(self, input_shape, hparams):
        # get backbone
        super(CLIPInhib, self).__init__()
        network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        n_outputs = network.num_features
        network.head = Identity()
        for name, param in network.named_parameters():
            param.requires_grad = False
        #require_grad_params1, names = inject_trainable_moe_kronecker_new(network, r=[1, 2, 4, 8], where='every_qkv')
        require_grad_params2 = inject_trainable_moe_kronecker_inhib(network, r=[1, 2, 4, 8], where='every_qkv')
        require_grad_params = require_grad_params2 # +require_grad_params1
        setattr(network, 'require_grad_params', require_grad_params)
        # set network and n oututs
        self.network, self.n_outputs = network, n_outputs
        print("Number of parameters in CLIP(base) original kadapter: ", sum(p.numel() for p in self.network.parameters()))
        print(self.network)

        #     
        self.blocknames = {  # clip vit-base
            "stem": ["patch_embed"],
            "block1": ["blocks.0", "blocks.1", "blocks.2"],
            "block2": ["blocks.3", "blocks.4", "blocks.5"],
            "block3": ["blocks.6", "blocks.7", "blocks.8"],
            "block4": ["blocks.9", "blocks.10", "blocks.11"],
        }
        self._features = []
        self.feat_layers = self.build_feature_hooks("block", self.blocknames)
        self.blocks = self.build_blocks(self.network, self.blocknames)


    def forward(self, x):
        self.clear_features()
        out = self.network(x)
        return out

    def clear_features(self):
        self._features.clear()




class DinoV2InhibBase(nn.Module):
    #
    #Vision Transformer with Attention Inhibition (AttentionIn) for parameter-efficient fine-tuning.
    
    #This class initializes the vit_small_inhib model, loads pre-trained weights excluding
    #adapter parameters, freezes the original model parameters, and ensures that only the
    #adapter parameters are trainable.
    
    def __init__(self, input_shape, hparams):
        super(DinoV2InhibBase, self).__init__()
        self.arch_name = "vit_base"
        self.patch_size = 14
        self.num_register_tokens = hparams.get('num_register_tokens', 0)
        print("Inhib type: AttentionInLateSoftmax")
        print("size:base")
        print("adapter_dim: 324")
        print("inhib_scale: self.inhib_scale ((self.adapter_dim//num_heads) ** -0.5)")
        print("proj_down:no")
        print("lambda_i initial: 0.05")
        print("softmax on attn_inhib: no")
        print("fine_tuning_mode: attn + adapter + lambda")
        # Initialize the vit_small_inhib model
        self.network = vit_base_inhib(
            patch_size=14,
            adapter_dim=324,
            inhib_type='AttentionInLateSoftmax',
            num_register_tokens=hparams.get('num_register_tokens', 0),
            img_size=518,
            init_values=1.0,
            block_chunks=0,
            )

        print("Number of parameters in DinoV2(s) Inhib: ", sum(p.numel() for p in self.network.parameters()))

        # Load pre-trained weights
        self._load_pretrained_weights()        
       
        # Define dropout and any additional layers if needed
        self.dropout = nn.Dropout(hparams.get('dropout', 0.1))
        
        # Define the number of output features (modify based on your requirements)
        self.n_outputs = 5 * 768

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'adapter' in n or 'lambda' in n or 'lora' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    
    def _load_pretrained_weights(self):
        #Loads pre-trained weights into the model, excluding adapter parameters.
        model_base_name = _make_dinov2_model_name(self.arch_name, self.patch_size)
        model_full_name = _make_dinov2_model_name(self.arch_name, self.patch_size, self.num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        pretrained_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.network.load_state_dict(pretrained_state_dict, strict=False)        
        print("Pre-trained weights loaded successfully, excluding adapter parameters.")
    
    def _freeze_original_parameters(self):
        
        #Freezes the original model parameters to prevent them from being updated during training.
        #Only adapter parameters remain trainable.
        
        for name, param in self.network.named_parameters():
            if not name.startswith('adapter'):
                param.requires_grad = False
        
        print("Original model parameters are frozen. Only adapter parameters are trainable.")
        
    def forward(self, x):
        
        #Forward pass of the VisionTransformerInhib model.
        
        #Args:
        #    x (Tensor): Input tensor of shape (B, C, H, W)
        
        #Returns:
        #    Tensor: Output features after processing through the network
        
        # Get intermediate layers
        intermediate_outputs = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        
        # Concatenate class tokens and mean of the last layer's tokens
        linear_input = torch.cat([
            intermediate_outputs[0][1],  # Class token from layer 1
            intermediate_outputs[1][1],  # Class token from layer 2
            intermediate_outputs[2][1],  # Class token from layer 3
            intermediate_outputs[3][1],  # Class token from layer 4
            intermediate_outputs[3][0].mean(1)  # Mean of tokens from layer 4
        ], dim=1)
        
        return self.dropout(linear_input)


class DinoV2Inhib(nn.Module):
    
    #Vision Transformer with Attention Inhibition (AttentionIn) for parameter-efficient fine-tuning.
    
    #This class initializes the vit_small_inhib model, loads pre-trained weights excluding
    #adapter parameters, freezes the original model parameters, and ensures that only the
    #adapter parameters are trainable.
    
    def __init__(self, input_shape, hparams):
        super(DinoV2Inhib, self).__init__()
        self.arch_name = "vit_small"
        self.patch_size = 14
        self.num_register_tokens = hparams.get('num_register_tokens', 0)
        print("Inhib type: Competitive adapter with qk")
        print("size:small")
        print("adapter_dim: 144")
        print("inhib_scale: self.inhib_scale ((self.adapter_dim//num_heads) ** -0.5)")
        print("proj_down:no")
        print("lambda_i initial: 0.05")
        print("softmax on attn_inhib: no")
        print("fine_tuning_mode: attn + adapter + lambda")
        # Initialize the vit_small_inhib model
        self.network = vit_small_inhib(
            patch_size=14,
            adapter_dim=144,
            inhib_type='AttentionIn',
            num_register_tokens=hparams.get('num_register_tokens', 0),
            img_size=518,
            init_values=1.0,
            block_chunks=0,
            )

        print("Number of parameters in DinoV2(s) Inhib: ", sum(p.numel() for p in self.network.parameters()))

        # Load pre-trained weights
        self._load_pretrained_weights()
        
        # Freeze original model parameters
        #self._freeze_original_parameters()
        
        # Define dropout and any additional layers if needed
        self.dropout = nn.Dropout(hparams.get('dropout', 0.1))
        
        # Define the number of output features (modify based on your requirements)
        self.n_outputs = 5 * 384

        if hparams["vit_attn_tune"]:
            for n,p in self.network.named_parameters():
                if 'adapter' in n or 'lambda' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    
    def _load_pretrained_weights(self):
        
        #Loads pre-trained weights into the model, excluding adapter parameters.
        
        model_base_name = _make_dinov2_model_name(self.arch_name, self.patch_size)
        model_full_name = _make_dinov2_model_name(self.arch_name, self.patch_size, self.num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        pretrained_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        #model_state_dict = self.network.state_dict()
        #pretrained_network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # # Filter out adapter parameters and only load matching parameters
        # filtered_pretrained_dict = {
        #     k: v for k, v in pretrained_state_dict.items()
        #     if k in model_state_dict and v.shape == model_state_dict[k].shape
        # }
        # # Update the model's state dict with the filtered pre-trained weights
        # model_state_dict.update(filtered_pretrained_dict)
        
        # Load the updated state dict into the model
        self.network.load_state_dict(pretrained_state_dict, strict=False)
        
        print("Pre-trained weights loaded successfully, excluding adapter parameters.")
    
    def _freeze_original_parameters(self):
        
        #Freezes the original model parameters to prevent them from being updated during training.
        #Only adapter parameters remain trainable.
        
        for name, param in self.network.named_parameters():
            if not name.startswith('adapter'):
                param.requires_grad = False
        
        print("Original model parameters are frozen. Only adapter parameters are trainable.")
        
    def forward(self, x):
        
        #Forward pass of the VisionTransformerInhib model.
        
        #Args:
        #    x (Tensor): Input tensor of shape (B, C, H, W)
        
        #Returns:
        #    Tensor: Output features after processing through the network
        
        # Get intermediate layers
        intermediate_outputs = self.network.get_intermediate_layers(x, n=4, return_class_token=True)
        
        # Concatenate class tokens and mean of the last layer's tokens
        linear_input = torch.cat([
            intermediate_outputs[0][1],  # Class token from layer 1
            intermediate_outputs[1][1],  # Class token from layer 2
            intermediate_outputs[2][1],  # Class token from layer 3
            intermediate_outputs[3][1],  # Class token from layer 4
            intermediate_outputs[3][0].mean(1)  # Mean of tokens from layer 4
        ], dim=1)
        
        return self.dropout(linear_input)

"""
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        if hparams['resnet50_augmix']:
            self.network = timm.create_model('resnet50.ram_in1k', pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        if hparams["freeze_bn"]:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["freeze_bn"]:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["vit"]:
            if hparams["dinov2"]:
                return DinoV2(input_shape, hparams)
            elif hparams["dinov2_inhib"]:
                return DinoV2Inhib(input_shape, hparams)
            elif hparams["dinov2_inhib_base"]:
                return DinoV2InhibBase(input_shape, hparams)
            elif hparams["clip_inhib"]:
                return CLIPInhib(input_shape, hparams)
            else:
                raise NotImplementedError
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
