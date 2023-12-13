import torch
import numpy as np
import pytorch_diffusion.diffusion as diff
from pytorch_diffusion.diffusion import Diffusion
from pytorch_diffusion.ckpt_util import get_ckpt_path
from pytorch_diffusion.diffusion import get_beta_schedule
import PIL
import torchvision.transforms as transforms
import sys, tqdm, PIL.Image

class DDRM(Diffusion):
    def __init__(self, diffusion_config, model_config, device=None):
        super().__init__(diffusion_config, model_config)
        self._logvar_ddrm()


    def _logvar_ddrm(self): 
      self.alphas_bar = np.cumprod(self.alphas, axis = 0)
      self.logvar_ddrm = torch.log(torch.Tensor((1 - self.alphas_bar) / self.alphas_bar))
      self.logvar_ddrm[0] = self.logvar[0]

    def denoise(self, n, y,
                degradation_model,
                hyperparams = dict(sigma_y = 0.1, eta = 0.85, eta_b = 1),
                n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i, x0=None: None):
        
        with torch.no_grad():
            if curr_step is None:
                curr_step = self.num_timesteps

            assert curr_step > 0, curr_step

            if n_steps is None or curr_step-n_steps < 0:
                n_steps = curr_step

            sigma_y = hyperparams["sigma_y"]
            eta = hyperparams["eta"]
            eta_b = hyperparams["eta_b"]

            xs = []
            if x is None:
                
                assert curr_step == self.num_timesteps, curr_step
                # Get y_bar from degradation model, shape Nx3x256x256
                y_bar = degradation_model.get_y_bar(y.to(self.device))#.view(y.shape)

                # Set t vector
                t = (torch.ones(n)*(curr_step-1)).to(self.device)

                # Set sigma_T_squared scalar value
                logvar_T = torch.Tensor([self.logvar_ddrm[-1]]).reshape(1,1,1,1).to(self.device)
                # logvar_T = torch.Tensor([self.logvar[-1]]).reshape(1,1,1,1).to(self.device)

                # Get singulars from degradation model
                # singulars = degradation_model.singulars.to(self.device).expand(3, degradation_model.singulars.shape[-1]).expand(y.shape[0],3,degradation_model.singulars.shape[-1]).view(y.shape)
                singulars = degradation_model.get_singulars(n).to(self.device)
                # singulars = singulars.view(y.shape)
                # Get sigma_y / singulars
                sigma_y_to_singulars_ratio = (sigma_y/singulars)

                # Get sigma_T_squared vector
                sigma_T_squared = torch.full_like(sigma_y_to_singulars_ratio, torch.exp(logvar_T).item()).to(self.device)

                # Initialise the vector for mean
                mean = y_bar
                # Set to 0 where singulars are zero
                mean[torch.where(singulars == 0)] = 0
                mean[torch.where(sigma_T_squared < sigma_y_to_singulars_ratio**2)] = 0
                # mean = mean  * torch.sqrt(1 + torch.exp(logvar_T))
                # Initialise the vector for variance
                variance = sigma_T_squared - sigma_y_to_singulars_ratio**2
                variance[torch.where(sigma_T_squared < sigma_y_to_singulars_ratio**2)] = sigma_T_squared[torch.where(sigma_T_squared < sigma_y_to_singulars_ratio)]
                # Set to sigma_T_squared where the singular values are zero
                variance[torch.where(singulars == 0)] = torch.exp(logvar_T).item()
                # variance = variance.view(mean.shape)
                # print(f"Mean shape {mean.shape} and variance shape {variance.shape}")
                # Sample from the distribution to get x_bar
                # x_bar = torch.normal(mean, std = torch.sqrt(variance))
                # x_bar = x_bar
                x_bar = torch.normal(mean, std = torch.sqrt(variance))

                # Get x from x_bar
                x = degradation_model.get_x_from_x_bar(x_bar).reshape(mean.shape).to(self.device) 
                x_m = x / torch.sqrt(1 + torch.exp(logvar_T))
                x_bar_m =  x_bar / torch.sqrt(1 + torch.exp(logvar_T))
                
                xs.append(x)
                x = x.to(self.device)
                step = int(self.num_timesteps/n_steps)
            for i in progress_bar(reversed(range(0, self.num_timesteps, step)), total=n_steps):
                # Get x_bar by performing denoising step
                x_bar, x0 = self.denoising_step(x = x_m,
                                       x_bar = x_bar,
                                       y_bar = y_bar,
                                       t=t,
                                       model=self.model,
                                       logvar=self.logvar,
                                       sigma_y = sigma_y,
                                       singulars = singulars,
                                       degradation_model = degradation_model,
                                       eta = eta,
                                       eta_b = eta_b,
                                       sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                       sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                       posterior_mean_coef1=self.posterior_mean_coef1,
                                       posterior_mean_coef2=self.posterior_mean_coef2,
                                       return_pred_xstart=True, 
                                       step = step)
                # Get x from x_bar
                x_bar_m = x_bar * torch.sqrt(torch.Tensor(diff.extract(self.alphas_bar, t - step, x.shape)))
                
                # x_m = degradation_model.get_x_from_x_bar(x_bar_m).reshape(x.shape).to(self.device) 
                # x = x_m * torch.sqrt(torch.Tensor(diff.extract(self.alphas_bar, t - 1, x.shape)))
                x =  degradation_model.get_x_from_x_bar(x_bar).reshape(x.shape).to(self.device) 
                x_m = x  * torch.sqrt(torch.Tensor(diff.extract(self.alphas_bar, t - step, x.shape)))

                # Set next t vector
                t = (torch.ones(n)*(i)).to(self.device)
                callback(x, i, x0=x0)
                xs.append(x)
            return x, xs


    def denoising_step(self, x, x_bar, y_bar, t, *,
                   model,
                   logvar,
                   sigma_y,
                   singulars,
                   degradation_model,
                   eta,
                   eta_b,
                   sqrt_recip_alphas_cumprod,
                   sqrt_recipm1_alphas_cumprod,
                   posterior_mean_coef1,
                   posterior_mean_coef2,
                   return_pred_xstart=True, 
                   step = 1):
        """
        Sample from p(x_{t-1} | x_t, y)
        """
        # Singulars reshape
        singulars = singulars.view(x.shape)
        sigma_T = torch.exp(torch.Tensor([self.logvar_ddrm[-1]]).reshape(1,1,1,1).to(self.device))
        # Get the output od the model
        model_output = model(x, t)

        # print(f"X shape {x.shape}")
        # Get the predicted x_start (x_0)
        # pred_xstart = (diff.extract(sqrt_recip_alphas_cumprod, t, x.shape)*x -
        #             diff.extract(sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
        alpha_bar_t_prev = torch.Tensor(diff.extract(self.alphas_bar, t, x.shape))
        pred_xstart = (x - torch.sqrt(1-alpha_bar_t_prev) * (model_output) )/ torch.sqrt(alpha_bar_t_prev) #/torch.sqrt(alpha_bar_t)
        pred_xstart = torch.clamp(pred_xstart, -1, 1)
        # Get x from x modified 
        x = x / torch.sqrt(alpha_bar_t_prev)
        # pred_x_t_mean = (diff.extract(posterior_mean_coef1, t, x.shape)*pred_xstart +
        #         diff.extract(posterior_mean_coef2, t, x.shape)*x).to(device=t.device)

        # logvariance_t_prev and sigma_t_prev
        logvar_t_prev = diff.extract(self.logvar_ddrm, t, x.shape)
        # sigma_t_prev = torch.full_like(singulars, torch.exp(0.5*logvar_t_prev)[0,0,0,0].item()).to(device=t.device)
        sigma_t_prev = torch.exp(0.5*logvar_t_prev).expand(singulars.shape)
        # Get x_start_bar
        pred_xstart_bar = degradation_model.get_x_bar(pred_xstart).view(singulars.shape)

        # Sigma_y vector
        sigma_y_vec = torch.full_like(singulars, sigma_y)
        # Sigma_y to singulars ratio
        sigma_y_to_singulars_ratio = sigma_y_vec/singulars

        # sigma_t - 1
        logvar_t = diff.extract(self.logvar_ddrm, t - step, x.shape)
        sigma_t = torch.full_like(singulars, torch.exp(0.5*logvar_t)[0,0,0,0].item()).to(device=t.device)

        # Eta squared
        eta_squared = eta**2

        # y bar
        y_bar = y_bar.to(device=t.device)

        def mean_singulars_zero(eta_squared, sigma_t, x_prev_bar, pred_xstart_bar, sigma_t_prev):
          one_minus_eta_sqrt = torch.sqrt(torch.tensor([1 - eta_squared])).to(self.device)
          return pred_xstart_bar + one_minus_eta_sqrt * sigma_t * (x_prev_bar - pred_xstart_bar) / sigma_t_prev

        def mean_sigma_t_lower(eta_squared, sigma_t, y_bar, pred_xstart_bar, sigma_y_to_singulars_ratio):
          one_minus_eta_sqrt = torch.sqrt(torch.tensor([1 - eta_squared])).to(self.device)
          return pred_xstart_bar + one_minus_eta_sqrt * sigma_t * (y_bar - pred_xstart_bar) / sigma_y_to_singulars_ratio

        def mean_sigma_t_higher(eta_b, sigma_t, pred_xstart_bar, y_bar):
          return (1 - eta_b) * pred_xstart_bar + eta_b * y_bar

        def variance_singulars_zero(eta_squared, sigma_t):
          return eta_squared * (sigma_t ** 2)

        def variance_sigma_t_higher(eta_b, sigma_t, sigma_y_to_singulars_ratio):
          return sigma_t**2 - sigma_y_to_singulars_ratio**2 * eta_b**2

        # First case sigma_t < sigma_y/singulars
        # print(f"Sigma lower {torch.sum(sigma_t < sigma_y_to_singulars_ratio)}")
        mean = mean_sigma_t_lower(eta_squared, sigma_t, y_bar, pred_xstart_bar, sigma_y_to_singulars_ratio)
        variance = variance_singulars_zero(eta_squared, sigma_t)
        # Second case higher
        mean[torch.where(sigma_t >= sigma_y_to_singulars_ratio)] = mean_sigma_t_higher(eta_b, sigma_t,  pred_xstart_bar, y_bar)[torch.where(sigma_t >= sigma_y_to_singulars_ratio)]
        variance[torch.where(sigma_t >= sigma_y_to_singulars_ratio)] = variance_sigma_t_higher(eta_b, sigma_t,  sigma_y_to_singulars_ratio)[torch.where(sigma_t >= sigma_y_to_singulars_ratio)]
        # Third case singulars are zeros
        mean[torch.where(singulars == 0)] = mean_singulars_zero(eta_squared, sigma_t, x_bar, pred_xstart_bar, sigma_t_prev)[torch.where(singulars == 0)]
        variance[torch.where(singulars == 0)] = variance_singulars_zero(eta_squared, sigma_t)[torch.where(singulars == 0)]

        # Sample x_bar from the distribution
        sample = torch.normal(mean, torch.sqrt(variance))
        sample = sample

        # print(f"Sample shape {sample.shape}")
        if return_pred_xstart:
            return sample, pred_xstart
        return sample

    @classmethod
    def from_pretrained(cls, name, device=None, num_timesteps=20, beta_end = 0.02, model_var_type = "fixedsmall"):
        cifar10_cfg = {
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,2,2,2),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.1,
        }
        lsun_cfg = {
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,1,2,2,4,4),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
        }
        image_net_cfg = {
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 256,
            "ch_mult": (1,1,2,2,4,4),
            "num_res_blocks": 2,
            "attn_resolutions": (32,16,8),
            "dropout": 0.0,
        }


        model_config_map = {
            "cifar10": cifar10_cfg,
            "lsun_bedroom": lsun_cfg,
            "lsun_cat": lsun_cfg,
            "lsun_church": lsun_cfg,
            "image_net": image_net_cfg
        }

        diffusion_config = {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": beta_end,
            "num_diffusion_timesteps": num_timesteps,
        }
        model_var_type_map = {
            "cifar10": "fixedlarge",
            "lsun_bedroom": "fixedsmall",
            "lsun_cat": "fixedsmall",
            "lsun_church": "fixedsmall",
        }
        ema = name.startswith("ema_")
        basename = name[len("ema_"):] if ema else name
        diffusion_config["model_var_type"] = model_var_type_map[basename] if model_var_type is None else model_var_type

        print("Instantiating")
        diffusion = cls(diffusion_config, model_config_map[basename], device)
        
        if name=="image_net":
           ckpt = "./checkpoints/256x256_diffusion_uncond.pt"
        else:
            ckpt = get_ckpt_path(name)
        print("Loading checkpoint {}".format(ckpt))
        diffusion.model.load_state_dict(torch.load(ckpt, map_location=diffusion.device), strict=False)
        # diffusion.model.load_state_dict(ckpt)
        diffusion.model.to(diffusion.device)
        diffusion.model.eval()
        print("Moved model to {}".format(diffusion.device))
        return diffusion

    @staticmethod
    def torch2hwcuint8(x, clip=False):
        if clip:
            x = torch.clamp(x, -1, 1)
        x = x.detach().cpu()
        x = x.permute(0,2,3,1)
        x = (x+1.0)*255.0
        print(x)
        x = x.numpy().astype(np.uint8)
        return x

    @staticmethod
    def save(x, format_string, start_idx=0):
        import os, PIL.Image
        os.makedirs(os.path.split(format_string)[0], exist_ok=True)
        x = DDRM.torch2hwcuint8(x)
        for i in range(x.shape[0]):

            PIL.Image.fromarray(x[i]).save(format_string.format(start_idx+i))