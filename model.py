import types
from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pytorch_lightning as pl
from lucent.modelzoo import inceptionv1

class AutoEncoder(nn.Module):
    def __init__(self, d_hidden, d_input, l1_coeff, device = None, seed = None):
        super(AutoEncoder, self).__init__()
        
        self.d_hidden = d_hidden
        self.d_input = d_input
        self.l1_coeff = l1_coeff
        
        if seed is not None:
            torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_input, self.d_hidden)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_hidden, self.d_input)))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(self.d_input))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        if device is not None:
            self.to(device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)

        self.config = {
            'd_hidden': d_hidden,
            'd_input': d_input,
            'l1_coeff': l1_coeff,
            'seed' : seed,
        }

    def forward(self, x):
        x_cent = x - self.b_dec.reshape(1, -1).repeat(x.shape[0], 1)
        acts = F.relu(x_cent @ self.W_enc + self.b_enc.reshape(1, -1).repeat(x.shape[0], 1))
        x_reconstruct = acts @ self.W_dec + self.b_dec.reshape(1, -1).repeat(x.shape[0], 1)
        l2_loss = torch.mean((x_reconstruct - x_cent) ** 2)
        l1_loss = torch.sum(torch.abs(acts)) * self.l1_coeff
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    def save(self):
        config = self.config
        os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
        with open(config['save_path'] + '.json', 'w') as f:
            json.dump(config, f)
        torch.save(self.state_dict(), config['save_path'] + '.pt')

    @classmethod
    def load(cls, model_id, path = None):
        with open('./MODELS.json', 'r') as f:
            models = json.load(f)
        if model_id in models:
            config = models[model_id]
            self = cls(config)
            self.load_state_dict(torch.load(config['save_path'] + '.pt'))
            return self
        else:
            try:
                config = json.load(open(path + '.json', 'r'))
                self = cls(config)
                self.load_state_dict(torch.load(path + '.pt'))
                return self
            except:
                raise Exception('Model not found')
            
    @classmethod
    def load_from_config(cls, config):
        self = cls(config)
        self.load_state_dict(torch.load(config['save_path'] + '.pt'))
        return self

def reshape_patches(x, patch_size):
    if patch_size is None:
        return x.mean(dim=(-1,-2))
    assert x.dim() == 4
    assert x.shape[-1]%patch_size == 0
    assert x.shape[-2]%patch_size == 0
    ## Averages over patches of size patch_size among the last two dimensions
    ## And then flatten the first 3 dimensions
    x = F.avg_pool2d(x, patch_size, stride = patch_size)
    x = x.split(1, dim=-1)
    x = [y.split(1, dim=-2) for y in x]
    x = [item for sublist in x for item in sublist]
    x = torch.cat(x, dim=0)
    return x.squeeze(-1).squeeze(-1)

class HookedModel(nn.Module):
    def __init__(self, model, layer_to_hook, **kwargs):
        super(HookedModel, self).__init__()
        self.model = model
        self.layer_to_hook = layer_to_hook
        if 'patch_size' in kwargs:
            self.patch_size = kwargs['patch_size']
        else:
            self.patch_size = None
        def hook_fn(m, i, o):
            self.activations = reshape_patches(o, self.patch_size)
        self.hook_fn = hook_fn
        self.activate_hook()
    def activate_hook(self):
        self.hook = self.layer_to_hook.register_forward_hook(self.hook_fn)
    def deactivate_hook(self):
        self.hook.remove()
    def __del__(self):
        self.deactivate_hook()
    def forward(self, x):
        _ = self.model(x)
        return self.activations
        

default_ae = AutoEncoder(1000, 2048, 0.01, seed = 0)
default_model = inceptionv1(pretrained=True)
default_model.eval()
default_hookedmodel = HookedModel(default_model, default_model.mixed4a)

class DictionnaryLearner(pl.LightningModule):
    def __init__(self, d_hidden = 10 * 508, d_input = 508, l1_coeff = 1e-2, seed = None, model_to_hook = 'inceptionv1', layer_to_hook = 'mixed4a', device = None, **kwargs):
        super().__init__()
        ## TODO: Add support for other models. Check that the inceptionv1 model is identical to the one used in torchhub.
        self.autoencoder = AutoEncoder(d_hidden, d_input, l1_coeff, device, seed)
        if 'patch_size' in kwargs:
            self.patch_size = kwargs['patch_size']
        self.hookedmodel = HookedModel(default_model, default_model.mixed4a, patch_size = self.patch_size)
        for param in self.hookedmodel.parameters():
            param.requires_grad = False
        self.save_hyperparameters()
        for key, value in kwargs.items():
            setattr(self, key, value)
    def forward(self, x):
        return self.autoencoder(x)
    def training_step(self, batch, batch_idx):
        x, _ = batch
        activations = self.hookedmodel(x)
        loss, _,_, l2_loss, l1_loss = self.autoencoder(activations)
        self.log('train_loss', loss)
        self.log('train_l2_loss', l2_loss)
        self.log('train_l1_loss', l1_loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        activations = self.hookedmodel(x)
        loss, _,_, l2_loss, l1_loss = self.autoencoder(activations)
        self.log('val_loss', loss)
        self.log('val_l2_loss', l2_loss)
        self.log('val_l1_loss', l1_loss)
        return loss
    def configure_optimizers(self):
        if hasattr(self, 'lr'):
            lr = self.lr
        else:
            lr = 1e-3
        if hasattr(self, 'optimizer'):
            if self.optimizer is not None:
                if isinstance(self.optimizer, str):
                    if self.optimizer == 'Adam':
                        return torch.optim.Adam(self.parameters(), lr=lr)
                    else:
                        raise NotImplementedError('Only Adam optimizer is supported for now')
                elif isinstance(self.optimizer, torch.optim.Optimizer):
                    return self.optimizer
                else:
                    raise ValueError('optimizer must be a string or a torch.optim.Optimizer or None')
        else:
            return torch.optim.Adam(self.parameters(), lr=lr)