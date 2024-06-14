from typing import Dict, Union
from einops import rearrange, repeat

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # print("additional_model_inputs", additional_model_inputs)
        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
            + input * c_skip
        )


class NVDenoiser(Denoiser):
    def __init__(
        self,
        depth: int,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.depth = depth # the depth of the latent

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor, # shape(torch.size[B,F,C,72,72]) here f = 1 for the first frame(input_depth)
        sigma: torch.Tensor,
        cond: Dict, # c = {"crossattn", shape(torch.size[1,1,1024]);"concat", shape(torch.size[1,4,72,72]; "vector", shape(torch.size[21,256]}
        **additional_model_inputs,
    ) -> torch.Tensor:
        
        bs, num_frames = input.shape[:2]
        assert num_frames == 1, f"num_frames = 1, becasue our input is just image, Expected num_frames to be 1, got {num_frames}"
        num_frames = self.depth # change here
        # input = rearrange(input, "b t ... -> (b t) ...") # rearrange for the input shape of network video_unet (b*f, c, h, w)
        
        # similar to the process in sampler
        for k in ["crossattn", "concat"]: 
            cond[k] = repeat(cond[k], "b ... -> b t ...", t=num_frames)
            # "crossattn" c[k] = shape(torch.size[bs,f(depth),1,1024])
            cond[k] = rearrange(cond[k], "b t ... -> (b t) ...", t=num_frames)
        
        additional_model_inputs = {}
        # TODO: The image_omly_indicator
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            2, num_frames
        ).to(input.device, input.dtype)
        additional_model_inputs["num_video_frames"] = num_frames
                    
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
