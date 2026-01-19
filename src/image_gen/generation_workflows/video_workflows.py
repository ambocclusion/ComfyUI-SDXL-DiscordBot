from PIL import Image

from comfy_script.runtime.nodes import *
from src.image_gen.generation_workflows.sd_workflows import SDWorkflow

class VideoWorkflow(SDWorkflow):
    def decode_and_save(self, file_name: str):
        self.controlnet_image = VAEDecode(Latent, self.vae)
        return VHSVideoCombine(self.image, self.params.fps, 0, "final_output", VHSVideoCombine.format.image_gif, False, True, None, None)

class SVDWorkflow(VideoWorkflow):
    def _load_model(self):
        self.model, self.clip_vision, self.vae = ImageOnlyCheckpointLoader(self.params.model)

    def create_img2img_latents(self, image_input: Image):
        with open(self.params.filename, "rb") as f:
            image = Image.open(f)
            width = image.width
            height = image.height
            padding = 0
            if width / height <= 1:
                padding = height // 2
        super().create_img2img_latents(image_input)
        self.image, _ = ImagePadForOutpaint(self.image, padding, 0, padding, 0, 40)
        # actual latents are created in condition_prompts

    def condition_prompts(self):
        self.model = VideoLinearCFGGuidance(Model, self.params.min_cfg)
        self.conditioning, self.negative_conditioning, self.latents = SVDImg2vidConditioning(
            self.clip_vision,
            self.image,
            self.vae,
            1024,
            576,
            25,
            self.params.motion,
            8,
            self.params.augmentation
        )
        if self.params.use_align_your_steps:
            self.scheduler = AlignYourStepsScheduler("SVD", self.params.num_steps)
            self.sampler = KSamplerSelect("euler")


class WANWorkflow(VideoWorkflow):
    pass


class WANImageWorkflow(VideoWorkflow):
    pass
