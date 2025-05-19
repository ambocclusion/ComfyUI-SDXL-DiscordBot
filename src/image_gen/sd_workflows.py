import dataclasses
from typing import Optional

from comfy_script.runtime import *
from src.image_gen.ImageWorkflow import *
from src.image_gen.controlnet_workflows import *
from src.util import get_server_address

load(get_server_address())

from comfy_script.runtime.nodes import *

controlnet_workflows = {
    ControlnetTypes.pose : PoseControlnetWorkflow,
    ControlnetTypes.depth : DepthControlnetWorkflow,
    ControlnetTypes.canny : CannyControlnetWorkflow,
}

@dataclasses.dataclass
class Lora:
    name: str
    strength: float


class SDWorkflow:
    def __init__(self, params: ImageWorkflow):
        self.params = params
        self._load_model()

    def _load_model(self):
        if self.params.use_tensorrt is False or self.params.tensorrt_model is None or self.params.tensorrt_model == "":
            model, clip, vae = CheckpointLoaderSimple(self.params.model)
        else:
            _, _, vae = CheckpointLoaderSimple(self.params.model)
            model = TensorRTLoader(self.params.tensorrt_model, TensorRTLoader.model_type.sd1_x if isinstance(self, SD15Workflow) else TensorRTLoader.model_type.sdxl_base)
        if self.params.vae is not None:
            vae = VAELoader(self.params.vae)
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name == None or lora.name == "None":
                    continue
                model, clip = LoraLoader(model, clip, lora.name, lora.strength, lora.strength)
        clip = CLIPSetLastLayer(clip, self.params.clip_skip)
        if self.params.controlnet_type is not None and self.params.controlnet_model is not None:
            self.controlnet_workflow = controlnet_workflows[ControlnetTypes[self.params.controlnet_type]](self.params)
        self.model = model
        self.clip = clip
        self.vae = vae
        self.clip_vision = None

    def create_latents(self):
        width, height = self.params.dimensions
        latent = EmptyLatentImage(width, height, self.params.batch_size)
        self.latents = [latent]

    def create_img2img_latents(self, image_input: Image):
        latent = VAEEncode(image_input, self.vae)
        if self.params.batch_size > 1:
            latent = RepeatLatentBatch(latent, self.params.batch_size)
        self.latents = [latent]

    def setup_for_animate_diff(self):
        context_options = ADEAnimateDiffUniformContextOptions(16, 2, 4, 'uniform', False, 'flat', False, 0, 1, None, None)
        motion_model_settings = ADEAnimateDiffModelSettingsSimple(0, None, 1, 1)
        self.model = ADEAnimateDiffLoaderWithContext(self.model, 'mm-Stabilized_mid.pth', 'sqrt_linear (AnimateDiff)', context_options, None, motion_model_settings, None, 1, False, None)

    def animate_diff_combine(self, images: Image):
        return VHSVideoCombine(images, 8, 0, 'final_output', 'image/gif', False, True, None, None)

    def condition_prompts(self):
        self.conditioning = CLIPTextEncode(self.params.prompt, self.clip)
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt or "", self.clip)
        if self.params.controlnet_type is not None and self.params.filename is not None:
            self.controlnet_image = self.controlnet_workflow.do_preprocessing(self.controlnet_image)
            self.conditioning, self.negative_conditioning = self.controlnet_workflow.do_conditioning(self.conditioning, self.negative_conditioning, self.controlnet_image, self.vae, self.params)

    def condition_for_detailing(self, controlnet_name: str, image: Image):
        pass

    def mask_for_inpainting(self, image_input: Image):
        clip_seg_model = CLIPSegModelLoader("CIDAS/clipseg-rd64-refined")
        masking, _ = CLIPSegMasking(image_input, self.params.inpainting_prompt, clip_seg_model)
        masking = MaskDominantRegion(masking, self.params.inpainting_detection_threshold)
        self.latents[0] = SetLatentNoiseMask(self.latents[0], masking)

    def unclip_encode(self, image_input: list[Image]):
        if self.clip_vision is None:
            self.clip_vision = CLIPVisionLoader(CLIPVisions.CLIP_ViT_bigG_14_laion2B_39B_b160k)
        for input in image_input:
            if input is None:
                continue
            self.model, self.ip_adapter = IPAdapterUnifiedLoader(self.model, IPAdapterUnifiedLoader.preset.STANDARD_medium_strength)
            self.model = IPAdapter(self.model, self.ip_adapter, input)
            encoded_clip_vision = CLIPVisionEncode(self.clip_vision, input)
            self.conditioning = UnCLIPConditioning(self.conditioning, encoded_clip_vision)

    def sample(self, use_ays: bool = False):
        if use_ays:
            num_steps = max(10, self.params.num_steps)
            sampler = KSamplerSelect(KSamplerSelect.sampler_name.dpmpp_2m_sde)
            model_type = AlignYourStepsScheduler.model_type.SDXL if isinstance(self, SDXLWorkflow) else AlignYourStepsScheduler.model_type.SD1
            sigmas = AlignYourStepsScheduler(model_type, num_steps, self.params.denoise_strength)
            self.output_latents, _ = SamplerCustom(self.model, True, self.params.seed, self.params.cfg_scale, self.conditioning, self.negative_conditioning, sampler, sigmas, self.latents[0])
        else:
            self.output_latents = KSampler(self.model, self.params.seed, self.params.num_steps, self.params.cfg_scale, self.params.sampler, self.params.scheduler or "normal", self.conditioning, self.negative_conditioning, self.latents[0], self.params.denoise_strength)

    def decode(self):
        return VAEDecode(self.output_latents, self.vae)

    def decode_and_save(self, file_name: str):
        image = VAEDecode(self.output_latents, self.vae)
        return SaveImage(image, file_name)


class SD15Workflow(SDWorkflow):
    pass


class SDXLWorkflow(SDWorkflow):
    def condition_prompts(self):
        self.conditioning = CLIPTextEncodeSDXL(self.clip, 4096, 4096, 0, 0, 4096, 4096, self.params.prompt, self.params.prompt)
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt, self.clip)
        if self.params.controlnet_type is not None and self.params.filename is not None:
            self.controlnet_image = LoadImage(self.params.filename)[0]
            self.conditioning, self.negative_conditioning = self.controlnet_workflow.do_conditioning(self.conditioning, self.negative_conditioning, self.controlnet_image, self.vae, self.params)

    def condition_for_detailing(self, controlnet_name: str, image: Image):
        if controlnet_name is None or controlnet_name == "":
            return
        try:
            image = TilePreprocessor(image, 1)
        except:
            print("no tile preprocessor")
        controlnet = ControlNetLoaderAdvanced(controlnet_name)
        self.conditioning, self.negative_conditioning, _ = ACNAdvancedControlNetApply(self.conditioning, self.negative_conditioning, controlnet, image, model_optional=self.model)

class PonyWorkflow(SDXLWorkflow):
    def condition_prompts(self):
        self.conditioning = CLIPTextEncodeSDXL(self.clip, 1024, 1024, 0, 0, 1024, 1024, self.params.prompt, self.params.prompt)
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt, self.clip)
        if self.params.controlnet_type is not None and self.params.filename is not None:
            self.controlnet_image = LoadImage(self.params.filename)[0]
            self.conditioning, self.negative_conditioning = self.controlnet_workflow.do_conditioning(self.conditioning, self.negative_conditioning, self.controlnet_image, self.vae, self.params)

class SDCascadeWorkflow(SDWorkflow):
    def _load_model(self):
        self.model, self.clip, self.stage_c_vae, self.clip_vision = UnCLIPCheckpointLoader(self.params.model)
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name == None or lora.name == "None":
                    continue
                self.model, self.clip = LoraLoader(self.model, self.clip, lora.name, lora.strength, lora.strength)
        self.stage_b_model, self.stage_b_clip, self.vae = CheckpointLoaderSimple(Checkpoints.cascade_stable_cascade_stage_b)

    def create_latents(self):
        width, height = self.params.dimensions
        latent_c, latent_b = StableCascadeEmptyLatentImage(width, height, 42, self.params.batch_size)
        self.latents = [latent_c, latent_b]

    def create_img2img_latents(self, image_input: Image):
        stage_c, stage_b = StableCascadeStageCVAEEncode(image_input, self.stage_c_vae, 32)
        stage_c = RepeatLatentBatch(stage_c, self.params.batch_size)
        stage_b = RepeatLatentBatch(stage_b, self.params.batch_size)
        self.latents = [stage_c, stage_b]

    def unclip_encode(self, image_input: list[Image]):
        for input in image_input:
            encoded_clip_vision = CLIPVisionEncode(self.clip_vision, input)
            self.conditioning = UnCLIPConditioning(self.conditioning, encoded_clip_vision)

    def condition_prompts(self):
        self.conditioning = CLIPTextEncode(self.params.prompt, self.clip)
        self.stage_c_conditioning = self.conditioning
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt or "", self.clip)

    def sample(self, use_ays: bool = False):
        stage_c = KSampler(self.model, self.params.seed, self.params.num_steps, self.params.cfg_scale, self.params.sampler, self.params.scheduler, self.conditioning, self.negative_conditioning, self.latents[0], self.params.denoise_strength)
        self.stage_b_conditioning = StableCascadeStageBConditioning(self.stage_c_conditioning, self.latents[0])
        conditioning2 = StableCascadeStageBConditioning(self.stage_c_conditioning, stage_c)
        zeroed_out = ConditioningZeroOut(self.stage_c_conditioning)
        self.output_latents = KSampler(self.stage_b_model, self.params.seed, 10, 1.1, self.params.sampler, self.params.scheduler, conditioning2, zeroed_out, self.latents[1], 1)

class SD3Workflow(SDWorkflow):
    def _load_model(self):
        if self.params.use_tensorrt is False or self.params.tensorrt_model is None or self.params.tensorrt_model == "":
            model, _, vae = CheckpointLoaderSimple(self.params.model)
        else:
            _, _, vae = CheckpointLoaderSimple(self.params.model)
            model = TensorRTLoader(self.params.tensorrt_model, TensorRTLoader.model_type.sd3)
        clip = TripleCLIPLoader(CLIPs.clip_l, CLIPs.clip_g, CLIPs.t5xxl_fp16)
        if self.params.vae is not None:
            vae = VAELoader(self.params.vae)
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name == None or lora.name == "None":
                    continue
                model, clip = LoraLoader(model, clip, lora.name, lora.strength, lora.strength)
        self.model = model
        self.clip = clip
        self.vae = vae
        self.clip_vision = None

    def create_latents(self):
        width, height = self.params.dimensions
        latent = EmptySD3LatentImage(width, height, self.params.batch_size)
        self.latents = [latent]

    def condition_prompts(self):
        self.conditioning = CLIPTextEncode(self.params.prompt, self.clip)
        self.negative_conditioning = CLIPTextEncode(self.params.negative_prompt or "", self.clip)
        zero_out_negative_conditioning = ConditioningZeroOut(self.negative_conditioning)
        negative_conditioning1 = ConditioningSetTimestepRange(zero_out_negative_conditioning, 0.1, 1)
        negative_conditioning2 = ConditioningSetTimestepRange(self.negative_conditioning, 0, 0.1)
        self.negative_conditioning = ConditioningCombine(negative_conditioning1, negative_conditioning2)

    def sample(self, use_ays: bool = False):
        self.model = ModelSamplingSD3(self.model, 3)
        super().sample(use_ays)

class FluxWorkflow(SDWorkflow):
    def _load_model(self):
        if self.params.model.endswith(".gguf"):
            model = UnetLoaderGGUF(self.params.model)
        elif "nf4" in self.params.model.lower():
            model, _, _ = CheckpointLoaderNF4(self.params.model)
        else:
            model = LoadDiffusionModel(self.params.model)
        clip_model = CLIPLoaderGGUF.clip_name.t5xxl_gguf if any(value.name.lower().endswith("gguf") for value in CLIPLoaderGGUF.clip_name) else CLIPs.t5xxl_fp16
        clip = DualCLIPLoaderGGUF(clip_model, CLIPs.clip_l, DualCLIPLoader.type.flux)
        if self.params.lora_dict:
            for lora in self.params.lora_dict:
                if lora.name == None or lora.name == "None":
                    continue
                model, clip = LoraLoader(model, clip, lora.name, lora.strength, lora.strength)
        if self.params.vae is not None:
            vae = VAELoader(self.params.vae)
        else:
            vae = VAELoader("ae.sft")
        if self.params.use_teacache is True and self.params.num_steps > 4:
            model = TeaCache(model, 'flux', 0.4, 3)
            if self.params.use_triton is True:
                model = CompileModel(model, 'default', 'inductor', False, False)
        width, height = self.params.dimensions
        model = ModelSamplingFlux(model,1.15, 0.5, width, height)
        if self.params.controlnet_type is not None and self.params.controlnet_model is not None:
            self.controlnet_image = LoadImage(self.params.filename)[0]
            self.controlnet_workflow = controlnet_workflows[ControlnetTypes[self.params.controlnet_type]](self.params)
        self.model = model
        self.clip = clip
        self.vae = vae

    def sample(self, use_ays: bool = False):
        self.conditioning = FluxGuidance(self.conditioning, self.params.cfg_scale)
        noise = RandomNoise(self.params.seed)
        guider = BasicGuider(self.model, self.conditioning)
        sampler = KSamplerSelect(self.params.sampler)
        sigmas = BasicScheduler(self.model, self.params.scheduler, self.params.num_steps, self.params.denoise_strength)
        self.output_latents, _ = SamplerCustomAdvanced(noise, guider, sampler, sigmas, self.latents[0])

    def unclip_encode(self, image_input: list[Image]):
        self.clip_vision = CLIPVisionLoader(CLIPVisions.sigclip_vision_patch14_384)

        style_model = StyleModelLoader(StyleModels.flux1_redux_dev)

        for i, input in enumerate(image_input):
            mashup_strength = self.params.mashup_image_strength if i == 0 else self.params.mashup_inputimage_strength
            if input is None:
                continue

            self.conditioning, _, _ = ReduxAdvanced(self.conditioning, style_model, self.clip_vision, input, 1, 'area', 'center crop (square)', mashup_strength)


class UpscaleWorkflow:
    def load_image(self, file_path: str):
        self.image, _ = LoadImage(file_path)

    def pass_image(self, image):
        self.image = image

    def upscale(self, model: str, rescale_factor: float):
        self.image, _ = CRUpscaleImage(self.image, model, 'rescale', rescale_factor, 1024, CRUpscaleImage.resampling_method.lanczos, True, 8)

    def save(self, file_path: str):
        return SaveImage(self.image, file_path)
