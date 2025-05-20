from comfy_script.runtime import *

from comfy_script.runtime.nodes import *
from src.image_gen.ImageWorkflow import ImageWorkflow
from src.util import get_server_address

load(get_server_address())

from comfy_script.runtime.nodes import *


class ControlnetWorkflow:
    def __init__(self, parameters: ImageWorkflow):
        if not parameters.controlnet_model:
            return
        self.controlnet_model = ControlNetLoader(parameters.controlnet_model)

    def do_preprocessing(self, image) -> Image:
        return None

    def do_conditioning(self, positive_conditioning, negative_conditioning, image, vae, parameters: ImageWorkflow) -> tuple[Conditioning, Conditioning]:
        if image is None:
            print("Error: No image provided for ControlNet conditioning")
            return positive_conditioning, negative_conditioning

        return ControlNetApplyAdvanced(positive_conditioning,
                                       negative_conditioning,
                                       self.controlnet_model,
                                       image,
                                       parameters.controlnet_strength,
                                       parameters.controlnet_start_percent,
                                       parameters.controlnet_end_percent,
                                       vae
                                       )


class PoseControlnetWorkflow(ControlnetWorkflow):
    def __init__(self, parameters: ImageWorkflow):
        super().__init__(parameters)
        self.controlnet_model = SetUnionControlNetType(self.controlnet_model, SetUnionControlNetType.type.openpose)

    def do_preprocessing(self, image) -> Image:
        if image is None:
            print("Error: No image provided for ControlNet preprocessing")
            return None
        image, _ = OpenposePreprocessor(image, resolution=512)
        return image


class CannyControlnetWorkflow(ControlnetWorkflow):
    def __init__(self, parameters: ImageWorkflow):
        super().__init__(parameters)
        self.controlnet_model = SetUnionControlNetType(self.controlnet_model, SetUnionControlNetType.type.canny_lineart_anime_lineart_mlsd)

    def do_preprocessing(self, image) -> Image:
        if image is None:
            print("Error: No image provided for ControlNet preprocessing")
            return None
        return Canny(image, low_threshold=0.15, high_threshold=0.3)


class DepthControlnetWorkflow(ControlnetWorkflow):
    def __init__(self, parameters: ImageWorkflow):
        super().__init__(parameters)
        self.controlnet_model = SetUnionControlNetType(self.controlnet_model, SetUnionControlNetType.type.depth)

    def do_preprocessing(self, image) -> Image:
        if image is None:
            print("Error: No image provided for ControlNet preprocessing")
            return None
        return DepthAnythingPreprocessor(image, resolution=512)
