import torch
import GlobVar as GV
import torch.nn as nn
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Materials
from typing import Optional
from pytorch3d.renderer.blending import (hard_rgb_blend,BlendParams)
from pytorch3d.renderer.mesh.rasterizer import (Fragments)
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.common.types import Device

class MaskRenderer(nn.Module):

    def __init__(
        self,
        device: Device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device: Device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = texels   
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images
    