import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"
NVWRAPPER = "sgm.modules.diffusionmodules.wrappers.NVWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        print("x.shape in OpenaiWrapper wrappers.py", x.shape)
        print("c_noise(timestep) in wrappers.py", t.shape)
        print("shape in wrappers.py", c.get("concat", None).shape, c.get("crossattn", None).shape), c.get("vector", None).shape
        assert x.shape == c.get("concat", None).shape
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1) # remove this concat, and we concat that
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None), # y is the vector
            **kwargs,
        )


class NVWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        print("x.shape in NVWrapper wrappers.py", x.shape)
        print("c_noise(timestep) in wrappers.py", t.shape) # TODO: should we add noise to 2D latent or 3D latent? Now we choose 2D latent
        print("c.shape in wrappers.py", c.get("concat", None).shape, c.get("crossattn", None).shape, c.get("vector", None).shape)
        # x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1) remove this concat, and we concat that
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            concat=c.get("concat", torch.Tensor([])), # here add the concat context becasue we remove the cat in wrapper
            y=c.get("vector", None), # y is the vector
            **kwargs,
        )