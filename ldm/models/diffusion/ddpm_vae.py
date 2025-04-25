"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.models.diffusion.ddpm import LatentDiffusion, DDPM


class DDPMVAE(LatentDiffusion):
    def __init__(self, *args, kl_weight=1., embeded=None, consistency_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.embeded = embeded
        self.consistency_weight = consistency_weight
        self.corrector = nn.Parameter(torch.tensor(0.))
        self.corrector_bias = nn.Parameter(torch.tensor(0.))

    def get_learned_conditioning(self, c, return_kl=False):
        c = self.cond_stage_model.encode(c)
        
        if isinstance(c, DiagonalGaussianDistribution):
            kl = c.kl()
            c = c.sample()
            c = rearrange(c, 'b c h w -> b (h w) c')
            
            if return_kl:
                return c, kl
        return c

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        c, kl = self.get_learned_conditioning(x, return_kl=True)
        loss, loss_dict, loss_simple = self.p_losses(x, c, t, return_simple=True, *args, **kwargs)

        if self.consistency_weight > 0:
            meanc = c.mean(dim=0, keepdim=True) * torch.ones_like(c)
            _, _, loss_simple2 = self.p_losses(x, meanc, t, return_simple=True, *args, **kwargs)

            scale = 10 * torch.nn.functional.softplus(self.corrector)
            ptrue = (loss_simple2 - loss_simple).unsqueeze(1) * scale
            loss_consistency = -nn.functional.logsigmoid(ptrue).mean()

            loss += self.consistency_weight * loss_consistency
            loss += self.consistency_weight * 0.01 * scale
            prefix = 'train' if self.training else 'val'
            loss_dict.update({f'{prefix}/consistency': loss_consistency})
            loss_dict.update({f'{prefix}/consistency_scale': scale.mean()})

        loss += self.kl_weight * kl.mean()
        prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{prefix}/kl': kl.mean()})

        return loss, loss_dict

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = DDPM.get_input(self, batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if self.embeded is not None:
            z = batch[self.embeded]
        else:
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()

        xc = z
        if force_c_encode:
            c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = xc
        if bs is not None:
            c = c[:bs]

        if self.use_positional_encodings:
            pos_x, pos_y = self.compute_latent_shifts(batch)
            ckey = __conditioning_keys__[self.model.conditioning_key]
            c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        params.append(self.corrector)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt