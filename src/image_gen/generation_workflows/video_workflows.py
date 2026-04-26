import os
import configparser
from math import floor

import PIL
from PIL import Image

from comfy_script.runtime import ImageBatchResult
from comfy_script.runtime.nodes import *
from src.image_gen.generation_workflows.image_workflows import SDWorkflow
from src.util import read_config, load_prompt_file

config = read_config()
comfy_root_directory = config["LOCAL"]["COMFY_ROOT_DIR"]

class VideoOutput(ImageBatchResult):
    def __init__(self, video_filenames):
        self.video_filenames = video_filenames
        self.files = PIL.Image.open(os.path.join(comfy_root_directory, "output", self.video_filenames["gifs"][0]["filename"]))
        super().__init__()

    async def get(self, index):
        return self.files.seek(index)

class VideoFile:
    """Wraps a saved MP4 file path for use in the image generation pipeline."""
    def __init__(self, path: str):
        self.path = path
        self.format = "MP4"


class VideoFileResult:
    """Awaitable wrapper for a VideoFile, compatible with the image batch pipeline."""
    def __init__(self, path: str):
        self._video_file = VideoFile(path)

    def __await__(self):
        return self._resolve().__await__()

    async def _resolve(self):
        return self

    async def get(self, index: int):
        return self._video_file


class VideoWorkflow(SDWorkflow):
    def decode_and_save(self, file_name: str):
        image = VAEDecode(self.output_latents, self.vae)
        self.output_images = SaveAnimatedWEBP(images=image, filename_prefix=file_name, fps=self.params.fps, method=SaveAnimatedWEBP.method.fastest, lossless=False)
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
        max_height = max_width * 0.5625
        self.image = ResizeImagesByLongerEdge(images=image_input, longer_edge=max_width)
        self.image = ResizeAndPadImage(image=self.image, target_width=max_width, target_height=max_height)
        self.latent = Wan22ImageToVideoLatent(self.vae, max_width, max_height, self.params.video_length, 1, self.image)

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


class LTXWorkflow(VideoWorkflow):
    # Default sigmas for LTX 2.3 two-pass distilled pipeline
    COARSE_SIGMAS = '1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'
    REFINE_SIGMAS = '0.85, 0.7250, 0.4219, 0.0'

    def _load_model(self):
        # UNET (GGUF or safetensors)
        if self.params.model.endswith('.gguf'):
            self.model = UnetLoaderGGUF(self.params.model)
        else:
            self.model = UNETLoader(self.params.model, 'default')

        # Distilled LoRA applied to model only (not CLIP)
        if self.params.use_accelerator_lora and self.params.accelerator_lora_name:
            self.model = LoraLoaderModelOnly(self.model, self.params.accelerator_lora_name, 0.6)

        # CLIP: Gemma text encoder + LTX text projection
        self.clip = DualCLIPLoader(self.params.clip_model, self.params.clip_model2, 'ltxv', 'default')

        # Apply user LoRAs to model + CLIP
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name is None or lora.name == 'None':
                    continue
                self.model, self.clip = LoraLoader(self.model, self.clip, lora.name, lora.strength, lora.strength)

        # Video VAE and audio VAE loaded separately
        self.vae = VAELoaderKJ(self.params.vae)
        self.audio_vae_model = VAELoaderKJ(self.params.audio_vae)

        # Spatial upscale model for two-pass pipeline
        self.upscale_model = LatentUpscaleModelLoader(self.params.latent_upscale_model)

        self._has_input_image = False
        self.encoded_audio = None

    def _get_dimensions(self):
        width = int(self.params.video_width) if self.params.video_width else 768
        width = (width // 32) * 32
        height = (width * 9 // 16 // 32) * 32
        return width, height

    def create_latents(self):
        width, height = self._get_dimensions()
        self.latent = EmptyLTXVLatentVideo(width, height, int(self.params.video_length), 1)
        self._has_input_image = False

    def create_img2img_latents(self, image_input: Image):
        width, height = self._get_dimensions()
        self.input_image = ResizeImagesByLongerEdge(images=image_input, longer_edge=width)
        self.input_image = ResizeAndPadImage(image=self.input_image, target_width=width, target_height=height)
        self.latent = EmptyLTXVLatentVideo(width, height, int(self.params.video_length), 1)
        self._has_input_image = True

    def enhance_prompt(self):
        if not self.params.enhance_ltx_prompt:
            return
        sampling_kwargs = {
            'sampling_mode.seed': self.params.seed or 0,
            'sampling_mode.temperature': 0.7,
            'sampling_mode.top_k': 64,
            'sampling_mode.top_p': 0.95,
            'sampling_mode.min_p': 0.05,
            'sampling_mode.repetition_penalty': 1.05,
        }
        if self.params.use_custom_system_prompt:
            stem = 'ltx_i2v_system_prompt' if self._has_input_image else 'ltx_t2v_system_prompt'
            system_prompt = load_prompt_file(stem)
            if self._has_input_image:
                formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>\n\nUser Raw Input Prompt: {self.params.prompt}.<end_of_turn>\n<start_of_turn>model\n"
            else:
                formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\nUser Raw Input Prompt: {self.params.prompt}.<end_of_turn>\n<start_of_turn>model\n"
            self._enhanced_prompt = TextGenerate(
                clip=self.clip,
                prompt=formatted_prompt,
                max_length=1024,
                sampling_mode='on',
                image=self.input_image if self._has_input_image else None,
                use_default_template=False,
                **sampling_kwargs,
            )
        else:
            self._enhanced_prompt = TextGenerateLTX2Prompt(
                clip=self.clip,
                prompt=self.params.prompt,
                max_length=1024,
                sampling_mode='on',
                image=self.input_image if self._has_input_image else None,
                use_default_template=True,
                **sampling_kwargs,
            )

    def condition_prompts(self):
        self.conditioning = CLIPTextEncode(self._get_prompt(), self.clip)
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt or '', self.clip)
        self.conditioning, self.negative_conditioning = LTXVConditioning(
            self.conditioning, self.negative_conditioning, float(self.params.fps or 24)
        )

        # Add input image as first-frame guide for img2video
        if self._has_input_image:
            self.conditioning, self.negative_conditioning, self.latent = LTXVAddGuide(
                self.conditioning, self.negative_conditioning, self.vae, self.latent,
                self.input_image, 0, 1.0
            )

        # Build audio latent: encode provided audio (with zero noise mask to preserve it),
        # or fall back to empty generated audio.
        if self.params.audio_filename:
            audio = LoadAudio(self.params.audio_filename)
            self.encoded_audio = LTXVAudioVAEEncode(audio, self.audio_vae_model)
            width, height = self._get_dimensions()
            mask = SolidMask(0.0, width, height)
            audio_latent = SetLatentNoiseMask(self.encoded_audio, mask)
        else:
            self.encoded_audio = None
            audio_latent = LTXVEmptyLatentAudio(
                frames_number=int(self.params.video_length),
                frame_rate=int(self.params.fps or 24),
                batch_size=1,
                audio_vae=self.audio_vae_model,
            )
        self.latent = LTXVConcatAVLatent(self.latent, audio_latent)

    def sample(self, use_ays: bool = False):
        # Pass 1: coarse sampling with euler_ancestral
        noise = RandomNoise(self.params.seed)
        guider = CFGGuider(self.model, self.conditioning, self.negative_conditioning, self.params.cfg_scale)
        coarse_out, _ = SamplerCustomAdvanced(
            noise, guider,
            KSamplerSelect('euler_ancestral'),
            ManualSigmas(self.COARSE_SIGMAS),
            self.latent,
        )

        # Separate AV, upscale video, crop guide keyframes
        video_latent, pass1_audio = LTXVSeparateAVLatent(coarse_out)
        video_latent = LTXVLatentUpsampler(video_latent, self.upscale_model, self.vae)
        positive2, negative2, _ = LTXVCropGuides(self.conditioning, self.negative_conditioning, video_latent)

        # Pass 2: refinement with euler on upscaled latent.
        # When audio was provided, use the original encoding as the guide rather than
        # the audio that passed through the coarse sampler.
        audio_for_pass2 = self.encoded_audio if self.encoded_audio is not None else pass1_audio
        noise2 = RandomNoise(self.params.seed)
        guider2 = CFGGuider(self.model, positive2, negative2, self.params.cfg_scale)
        combined = LTXVConcatAVLatent(video_latent, audio_for_pass2)
        _, refined_out = SamplerCustomAdvanced(
            noise2, guider2,
            KSamplerSelect('euler'),
            ManualSigmas(self.REFINE_SIGMAS),
            combined,
        )

        # Separate final video latent for decoding.
        # When audio was provided, use the original VAE-encoded audio for output rather
        # than the sampler's output — the refine sigmas would otherwise regenerate it.
        self.output_latents, sampled_audio = LTXVSeparateAVLatent(refined_out)
        self.audio_latents = self.encoded_audio if self.encoded_audio is not None else sampled_audio

    def decode_and_save(self, file_name: str):
        image = VAEDecodeTiled(self.output_latents, self.vae, tile_size=512, overlap=64, temporal_size=2048, temporal_overlap=8)
        audio = LTXVAudioVAEDecode(samples=self.audio_latents, audio_vae=self.audio_vae_model)
        video = CreateVideo(images=image, audio=audio, fps=int(self.params.fps or 24))
        self.output_images = SaveVideo(video=video, filename_prefix=file_name)
        return self.output_images

    async def wait_for_result(self):
        # Do NOT await the result — ImageBatchResult.__await__ calls PIL.Image.open on the
        # downloaded bytes, which fails for MP4. The job is already complete after the first
        # await, so read the filename directly from the output dict.
        result = await self.output_images
        entry = result._output['images'][0]
        mp4_path = os.path.join(comfy_root_directory, "output", entry.get('subfolder', ''), entry['filename'])
        return VideoFileResult(mp4_path)
