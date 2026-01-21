from comfy_script.runtime.nodes import Image
from src.image_gen.ImageWorkflow import ImageWorkflow


class GenerationWorkflow:
    def __init__(self, params: ImageWorkflow):
        self.params = params
        self._load_model()
    
    def _load_model(self):
        pass
    
    def create_latents(self):
        pass
    
    def create_img2img_latents(self, image_input: Image):
        pass
    
    def condition_prompts(self):
        pass
    
    def do_controlnet_workflow(self):
        pass
    
    def do_controlnet_conditioning(self):
        pass
    
    def should_do_controlnet(self):
        pass
    
    def condition_for_detailing(self, controlnet_name: str, image: Image):
        pass
    
    def mask_for_inpainting(self, image_input: Image):
        pass

    def unclip_encode(self, image_input: list[Image]):
        pass
    
    def sample(self, use_ays: bool = False):
        pass
    
    def decode(self):
        pass
    
    def decode_and_save(self, file_name: str):
        pass
    
    def resize_edit_image(self, image_input: Image) -> Image:
        pass
    
    async def wait_for_result(self):
        pass