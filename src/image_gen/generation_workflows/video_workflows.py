import os
import configparser
from math import floor

import PIL
from PIL import Image

from comfy_script.runtime import ImageBatchResult
from comfy_script.runtime.nodes import *
from src.image_gen.generation_workflows.image_workflows import SDWorkflow
from src.util import read_config

config = read_config()
comfy_root_directory = config["LOCAL"]["COMFY_ROOT_DIR"]

class VideoOutput(ImageBatchResult):
    def __init__(self, video_filenames):
        self.video_filenames = video_filenames
        self.files = PIL.Image.open(os.path.join(comfy_root_directory, "output", self.video_filenames["gifs"][0]["filename"]))
        super().__init__()

    async def get(self, index):
        return self.files.seek(index)

class VideoWorkflow(SDWorkflow):
    def decode_and_save(self, file_name: str):
        image = VAEDecode(self.output_latents, self.vae)
        self.output_images = SaveAnimatedWEBP(images=image, filename_prefix=file_name, fps=self.params.fps)
        return self.output_images
    

class SVDWorkflow(VideoWorkflow):
    def _load_model(self):
        self.model, self.clip_vision, self.vae = ImageOnlyCheckpointLoader(self.params.model)

    def create_img2img_latents(self, image_input: Image):
        with open(self.params.filename, "rb") as f:
            image = PIL.Image.open(f)
            width = image.width
            height = image.height
            padding = 0
            if width / height <= 1:
                padding = height // 2
        self.image, _ = ImagePadForOutpaint(image_input, padding, 0, padding, 0, 40)
        # actual latents are created in condition_prompts

    def condition_prompts(self):
        self.model = VideoLinearCFGGuidance(self.model, self.params.min_cfg)
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

    def sample(self, use_ays: bool = False):
        if use_ays:
            self.scheduler = AlignYourStepsScheduler("SVD", self.params.num_steps)
            self.sampler = KSamplerSelect("euler")
            self.output_latents, _ = SamplerCustom(self.model, True, self.params.seed, self.params.cfg_scale, self.conditioning, self.negative_conditioning, self.sampler, self.scheduler, self.latents)
        else:
            self.output_latents = KSampler(self.model,
                                           self.params.seed,
                                           self.params.num_steps,
                                           self.params.cfg_scale,
                                           self.params.sampler,
                                           self.params.scheduler,
                                           self.conditioning,
                                           self.negative_conditioning,
                                           self.latents,
                                           1
                                           )


class WANWorkflow(VideoWorkflow):
    def _load_model(self):
        if self.params.model.endswith(".gguf"):
            self.model = UnetLoaderGGUF(self.params.model)
        else:
            self.model = UNETLoader(self.params.model)
        self.clip_model = self.params.clip_model
        self.clip = CLIPLoaderGGUF(self.clip_model, "wan")
        self.vae = VAELoader(self.params.vae)
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name == None or lora.name == "None":
                    continue
                self.model, self.clip = LoraLoader(self.model, self.clip, lora.name, lora.strength, lora.strength)
        if self.params.use_accelerator_lora == "true":
            self.model_distilled = LoraLoaderModelOnly(self.model, self.params.accelerator_lora_name, 1)
        if self.params.use_teacache == "true":
            if "14B" in self.params.model:
                self.model = EasyCache(self.model, 0.05, 0.15, 0.95, True)
            else:
                self.model = EasyCache(self.model, 0.15, 0.25, 0.95, True)
        if self.params.use_triton == "true":
            self.model = ModelCompile(self.model)

    def create_latents(self):
        aspect_ratio = 1.77
        width = self.params.video_width
        height = floor(width / aspect_ratio)
        self.latent = Wan22ImageToVideoLatent(self.vae, width, height, self.params.video_length, 1)

    def create_img2img_latents(self, image_input: Image):
        max_width = self.params.video_width
        output_path = ''
        with open(self.params.filename, "rb") as f:
            self.image = PIL.Image.open(f)
            width = self.image.width
            height = self.image.height
            if self.params.crop_image or width > max_width or height > max_width:
                from src.imgutils import smart_crop, smart_resize
                self.image = smart_crop(self.image)
                self.image = smart_resize(self.image, max_width)
                # Save the resized image
                output_path, filename = os.path.split(self.params.filename)
                new_filename = f"wan_{filename}"
                output_path = output_path + "/" + new_filename
                self.image.save(fp=output_path)
            width = self.image.width
            height = self.image.height
        if output_path != '':
            self.image = LoadImage(output_path)[0]
        else:
            self.image = LoadImage(self.params.filename)[0]
        self.latent = Wan22ImageToVideoLatent(self.vae, width, height, self.params.video_length, 1, self.image)

    def sample(self, use_ays: bool = False):
        if self.params.use_accelerator_lora == "true":
            self.output_latents = KSamplerAdvanced(self.model,
                                           'enable',
                                           self.params.seed,
                                           self.params.num_steps,
                                           self.params.cfg_scale,
                                           self.params.sampler,
                                           self.params.scheduler,
                                           self.conditioning,
                                           self.negative_conditioning,
                                           self.latent,
                                           0,
                                           10,
                                           'enable'
                                           )
            self.output_latents = KSamplerAdvanced(self.model_distilled,
                                           'disable',
                                           0,
                                           self.params.num_steps,
                                           1,
                                           'gradient_estimation',
                                           'normal',
                                           self.conditioning,
                                           self.conditioning,
                                           self.latent,
                                           10,
                                           1000,
                                           'disable'
                                           )
        else:
            self.output_latents = KSampler(self.model,
                                   self.params.seed,
                                   self.params.num_steps,
                                   self.params.cfg_scale,
                                   self.params.sampler,
                                   self.params.scheduler,
                                   self.conditioning,
                                   self.negative_conditioning,
                                   self.latent,
                                   1.0
                                   )

    def condition_prompts(self):
        self.model = ModelSamplingSD3(self.model)
        super().condition_prompts()
