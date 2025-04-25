import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import os, sys
from torchvision.utils import make_grid
from torchvision.transforms import v2
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # Updated import location

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddpm import LatentDiffusion


def disabled_train(self, mode=True):
    """Overwrite model.train with this function so that train/eval mode doesn't change."""
    return self


class LatentVAE(LatentDiffusion):
    """Main class updated for PyTorch Lightning 2.x"""
    def __init__(self,
                 vae_config,
                 first_stage_config,
                 lossconfig,
                 use_ema=True,
                 first_stage_key="image",
                 scale_factor=1.0,
                 scale_by_std=False,
                 monitor="val/loss",
                 scheduler_config=None,
                 data_root=None,
                 embeded=None,
                 *args, **kwargs):
        pl.LightningModule.__init__(self)  # Use Python's super() for initialization
        
        self.scale_by_std = scale_by_std
        # for backwards compatibility after implementation of DiffusionWrapper
        self.conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.cond_stage_trainable = False
        self.cond_stage_key = None
        self.first_stage_key = first_stage_key
        self.use_ema = use_ema
        self.data_root = data_root
        self.embeded = embeded
        self.use_positional_encodings = False
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.model = instantiate_from_config(vae_config)
        self.model.conditioning_key = None
        self.instantiate_first_stage(first_stage_config)
        self.loss = instantiate_from_config(lossconfig)

        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        if monitor is not None:
            self.monitor = monitor
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
        self.save_hyperparameters()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        first_stage_model = model.eval()
        first_stage_model.train = disabled_train
        for param in first_stage_model.parameters():
            param.requires_grad = False
        # Use torch.compile if available (PyTorch 2.x) for potential speedups
        self.first_stage_model = first_stage_model

    def forward(self, x, c=None, *args, **kwargs):
        return self.model(x)

    def shared_step(self, batch, split="train"):
        if self.embeded is not None:
            inputs = batch[self.embeded]
        else:
            inputs, _ = self.get_input(batch, self.first_stage_key)
            if self.data_root:
                for i, rp in enumerate(batch['relpath']):
                    os.makedirs(os.path.dirname(os.path.join(self.data_root, rp.split('.')[0])), exist_ok=True)
                    np.savez_compressed(os.path.join(self.data_root, rp.split('.')[0]), inputs=inputs[i].cpu().numpy())

        reconstructions, posterior = self(inputs)
        return self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                         last_layer=None, split=split)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, split="train")
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            # Access the current optimizer via self.trainer.optimizers (assumes single optimizer)
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, split="val")
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, split="val")
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters()) + list(self.loss.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler_inst = instantiate_from_config(self.scheduler_config)
            # Here we set up a LambdaLR scheduler using the provided lambda function
            scheduler = {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler_inst.schedule),
                'interval': 'step',
                'frequency': 1
            }
            return [opt], [scheduler]
        return opt

    @torch.no_grad()
    def sample(self, cond=None, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        return self.model.sample(batch_size=batch_size)

    @torch.no_grad()
    def sample_log(self, batch_size, **kwargs):
        samples = self.sample(batch_size=batch_size, **kwargs)
        return samples, None

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, return_keys=None,
                   quantize_denoised=True, **kwargs):
        log = dict()
        # Retrieve inputs and first-stage outputs; get_input and decode_first_stage should be defined elsewhere.
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                            return_first_stage_outputs=True,
                                            force_c_encode=True,
                                            return_original_cond=True,
                                            bs=N)
        
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["first_stage_reconstruction"] = xrec
        log["reconstruction"] = self.decode_first_stage(self(z)[0])

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, _ = self.sample_log(batch_size=N)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, _ = self.sample_log(batch_size=N)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if return_keys:
            # Return only the requested keys if they exist in the log.
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


class IterativeAmortized(LatentVAE):
    def __init__(self, *args, vae_config, iterations=2, eps_only=True, **kwargs):
        in_channels = vae_config.params['ddconfig']['in_channels']
        z_channels = vae_config.params['ddconfig']['z_channels']
        vae_config.params['ddconfig']['in_channels'] = 3 * in_channels if eps_only else 4 * in_channels
        super().__init__(*args, vae_config=vae_config, **kwargs)
        self.down_factor = 2 ** (len(vae_config.params['ddconfig']['ch_mult']) - 1)
        self.post_upscale = nn.ConvTranspose2d(2 * z_channels, 2 * in_channels, kernel_size=self.down_factor, stride=self.down_factor, padding=0)
        self.iterations = iterations
        self.z_channels = z_channels
        self.eps_only = eps_only

    def forward(self, x, eps=None, post=None, *args, **kwargs):
        if eps is None:
            eps = x
        if post is None:
            post = torch.zeros(x.shape[0], 2 * self.z_channels, x.shape[2] // self.down_factor, x.shape[3] // self.down_factor, device=x.device)
        post_input = self.post_upscale(post)
        x = torch.cat([post_input, eps], dim=1) if self.eps_only else torch.cat([x, post_input, eps], dim=1)
        posterior = self.model.encode(x)
        z = posterior.sample()
        reconstructions = self.model.decode(z)
        return reconstructions, posterior

    def shared_step(self, batch, split="train"):
        if self.embeded is not None:
            inputs = batch[self.embeded]
        else:
            inputs, _ = self.get_input(batch, self.first_stage_key)

        loss, loss_dict = 0, {}
        eps, post = None, None
        for i in range(self.iterations):
            reconstructions, posterior = self(inputs, eps, post)
            lossi, loss_dicti =  self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                            last_layer=None, split=split)
            for key in loss_dicti:
                loss_dict[f"{key}_{i}"] = loss_dicti[key]
            loss += lossi
            eps = inputs - reconstructions
            post = posterior.parameters
        return loss, loss_dict
    
class Inpainting(LatentVAE):
    def __init__(self, *args, vae_config, post_match=True, pm_weight=1.0, **kwargs):
        in_channels = vae_config.params['ddconfig']['in_channels']
        z_channels = vae_config.params['ddconfig']['z_channels']
        vae_config.params['ddconfig']['in_channels'] = 2 * in_channels
        super().__init__(*args, vae_config=vae_config, **kwargs)
        self.eraser = v2.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0, inplace=False)
        self.post_match = post_match
        self.pm_weight = pm_weight

    def forward(self, x, mask=None, reconstruct=True, *args, **kwargs):
        if mask is None:
            mask = torch.ones_like(x)
        x = x * mask
        x = torch.cat([x, 1 - mask], dim=1)
        posterior = self.model.encode(x)
        z = posterior.sample()
        reconstructions = self.model.decode(z) if reconstruct else None
        return reconstructions, posterior

    def shared_step(self, batch, split="train"):
        if self.embeded is not None:
            inputs = batch[self.embeded]
        else:
            inputs, _ = self.get_input(batch, self.first_stage_key)
        mask = torch.ones_like(inputs)
        mask = self.eraser(mask)

        reconstructions, posterior = self(inputs, mask=None if self.post_match else mask)
        loss, loss_dict = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                         last_layer=None, split=split)
        
        if self.post_match:
            _, pm_posterior = self(inputs, mask=mask, reconstruct=False)
            pm_loss = DiagonalGaussianDistribution(posterior.parameters.detach()).kl(pm_posterior)
            pm_loss = pm_loss.sum() / pm_loss.shape[0]
            loss += self.pm_weight * pm_loss
            loss_dict["{}/pm_loss".format(split)] = pm_loss.detach().mean()
        return loss, loss_dict
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = super().log_images(batch, **kwargs)
        inputs, _ = self.get_input(batch, self.first_stage_key)
        mask = torch.ones_like(inputs)
        mask = self.eraser(mask)
        log["masked_reconstruction"] = self.decode_first_stage(self(inputs, mask=mask)[0])
        log["masked"] = self.decode_first_stage(inputs * mask)
        return log
    
class Perceptual(LatentVAE):
    def get_last_layer(self):
        return self.model.decoder.conv_out.weight

    def training_step(self, batch, batch_idx):
        inputs, _ = self.get_input(batch, self.first_stage_key)
        reconstructions, posterior = self(inputs)


        lossrg = []
        for p in self.loss.parameters():
            p.requires_grad = False
            lossrg.append(p.requires_grad)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        for p, rg in zip(self.loss.parameters(), lossrg):
            p.requires_grad = rg

        modelrg = []
        for p in self.model.parameters():
            p.requires_grad = False
            modelrg.append(p.requires_grad)
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
        for p, rg in zip(self.model.parameters(), modelrg):
            p.requires_grad = rg

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss + discloss
    
    def validation_step(self, batch, batch_idx):
        inputs, _ = self.get_input(batch, self.first_stage_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


        
