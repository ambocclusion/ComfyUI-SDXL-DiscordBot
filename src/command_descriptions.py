import json

from discord.app_commands import Choice

from src.comfyscript_utils import get_models, get_loras, get_samplers, get_schedulers
from src.consts import *
from src.defaults import *

models = get_models()
loras = get_loras()
samplers = get_samplers()
schedulers = get_schedulers()

generation_messages = json.loads(open("./data/generation_messages.json", "r").read())
completion_messages = json.loads(open("./data/completion_messages.json", "r").read())

# These aspect ratio resolution values correspond to the SDXL Empty Latent Image node.
# A latent modification node in the workflow converts it to the equivalent SD 1.5 resolution values.
ASPECT_RATIO_CHOICES = [
    Choice(name="1:1", value="1:1"),
    Choice(name="3:4 portrait", value="3:4 portrait"),
    Choice(name="9:16 portrait", value="9:16 portrait"),
    Choice(name="4:3 landscape", value="4:3 landscape"),
    Choice(name="16:9 landscape", value="16:9 landscape"),
]


def should_filter_model(m, command):
    if "hidden" in m.lower():
        return True
    if "lightning" in m.lower():
        return True
    if "turbo" in m.lower():
        return True
    if command != "sdxl" and "xl" in m.lower():
        return True
    if command == "sdxl" and "xl" not in m.lower():
        return True
    if "refiner" in m.lower():
        return True
    if command.lower() != "sdxl" and command.lower() not in m.lower():
        return True
    return False


SD15_MODEL_CHOICES = [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, "15")]
SD15_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "15")]
SDXL_MODEL_CHOICES = [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, "sdxl")]
SDXL_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "sdxl")]
PONY_MODEL_CHOICES = [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, "pony")]
PONY_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "pony")]
CASCADE_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "cascade")]
SD3_MODEL_CHOICES = [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, "sd3")]
SD3_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "sd3")]
FLUX_MODEL_CHOICES = [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, "flux")]
FLUX_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "flux")]
VIDEO_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "wan")]
SAMPLER_CHOICES = [Choice(name=s, value=s) for s in samplers if "adaptive" not in s.lower()]
SCHEDULER_CHOICES = [Choice(name=s, value=s) for s in schedulers]

COMMAND_MODEL_CHOICES = {
    "sdxl": SDXL_MODEL_CHOICES,
    "legacy": SD15_MODEL_CHOICES,
    "pony": PONY_MODEL_CHOICES,
}

COMMAND_LORA_CHOICES = {
    "sdxl": SDXL_LORA_CHOICES,
    "legacy": SD15_LORA_CHOICES,
    "pony": PONY_LORA_CHOICES,
    "cascade": CASCADE_LORA_CHOICES,
}

BASE_ARG_DESCS = {
    "prompt": "Prompt for the image being generated",
    "negative_prompt": "Prompt for what you want to steer the AI away from",
}
IMAGE_GEN_DESCS = {
    "model": "Model checkpoint to use",
    "lora": "LoRA to apply",
    "lora_strength": "Strength of LoRA",
    "aspect_ratio": "Aspect ratio of the generated image",
    "sampler": "Sampling algorithm to use",
    "num_steps": f"range [1, {MAX_STEPS}]; Number of sampling steps",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should follow prompt",
    "scheduler": "Changes which noise scheduler applies to the image. Use in conjunction with sampler",
}
IMAGINE_ARG_DESCS = {
    **BASE_ARG_DESCS,
    **IMAGE_GEN_DESCS,
    "num_steps": "Number of sampling steps; range [1, 30]",
    "input_file": "Image to use as input for img2img",
    "denoise_strength": f"range [0.01, 1.0], default {SD15_GENERATION_DEFAULTS.denoise_strength}; Strength of denoising filter during img2img. Only works when input_file is set",
    "inpainting_prompt": "Detection prompt for inpainting; examples: 'background' or 'person'",
    "inpainting_detection_threshold": f"range [0, 255], default {SD15_GENERATION_DEFAULTS.inpainting_detection_threshold}; Detection threshold for inpainting. Only works when inpainting_prompt is set",
    "clip_skip": f"default: {SD15_GENERATION_DEFAULTS.clip_skip}",
}
SDXL_ARG_DESCS = {
    **BASE_ARG_DESCS,
    **IMAGE_GEN_DESCS,
    "input_file": "Image to use as input for img2img",
    "denoise_strength": f"range [0.01, 1.0], default {SDXL_GENERATION_DEFAULTS.denoise_strength}; Strength of denoising filter during img2img. Only works when input_file is set",
    "inpainting_prompt": "Detection prompt for inpainting; examples: 'background' or 'person'",
    "inpainting_detection_threshold": f"range [0, 255], default {SDXL_GENERATION_DEFAULTS.inpainting_detection_threshold}; Detection threshold for inpainting. Only works when inpainting_prompt is set",
    "clip_skip": f"default: {SDXL_GENERATION_DEFAULTS.clip_skip}",
}
SVD_ARG_DESCS = {
    "input_file": "Starting image for video generation",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should adhere to the starting image. Default: {SVD_GENERATION_DEFAULTS.cfg_scale}",
    "min_cfg": f"Starting CFG value. Generation will move to CFG_SCALE over the length of the video. Default: {SVD_GENERATION_DEFAULTS.min_cfg}",
    "motion": f"The amount of motion in the video. Default: {SVD_GENERATION_DEFAULTS.motion}",
    "augmentation": f"How much the video will differ from your starting image. Introduces a lot of noise. Default: {SVD_GENERATION_DEFAULTS.augmentation}",
}
VIDEO_ARG_DESCS = {
    "prompt": "Prompt for the video being generated",
    "negative_prompt": "Prompt for what you want to steer the AI away from",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should follow prompt",
    "input_file": "Image to use as input",
    "lora": "LoRA to apply",
}
CASCADE_ARG_DESCS = {
    **BASE_ARG_DESCS,
    "lora": "LoRA to apply",
    "lora_strength": "Strength of LoRA",
    "aspect_ratio": "Aspect ratio of the generated image",
    "num_steps": f"range [1, {MAX_STEPS}]; Number of sampling steps",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should follow prompt",
    "input_file": "Image to use as input for img2img",
    "mashup_image": "Image to use as reference for a mashup. If input_file is set it will mash these two images together!",
    "denoise_strength": f"range [0.01, 1.0], default {CASCADE_GENERATION_DEFAULTS.denoise_strength}; Strength of denoising filter during img2img. Only works when input_file is set",
    "inpainting_prompt": "Detection prompt for inpainting; examples: 'background' or 'person'",
    "inpainting_detection_threshold": f"range [0, 255], default {CASCADE_GENERATION_DEFAULTS.inpainting_detection_threshold}; Detection threshold for inpainting. Only works when inpainting_prompt is set",
    "clip_skip": f"default: {CASCADE_GENERATION_DEFAULTS.clip_skip}",
}

PONY_ARG_DESCS = {
    "style_prompt": f"Aesthetic tags, default: {PONY_GENERATION_DEFAULTS.style_prompt}",
    "negative_style_prompt": f"Negative aesthetic tags, default: {PONY_GENERATION_DEFAULTS.negative_style_prompt}",
    **SDXL_ARG_DESCS
}

SD3_ARG_DESCS = {
    **BASE_ARG_DESCS,
}

FLUX_ARG_DESCS = {
    **BASE_ARG_DESCS,
}

BASE_ARG_CHOICES = {
    "aspect_ratio": ASPECT_RATIO_CHOICES[:25],
    "sampler": SAMPLER_CHOICES[:25],
    "scheduler": SCHEDULER_CHOICES[:25]
}
IMAGINE_ARG_CHOICES = {
    "model": SD15_MODEL_CHOICES[:25],
    "lora": SD15_LORA_CHOICES[:25],
    "lora2": SD15_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
SDXL_ARG_CHOICES = {
    "model": SDXL_MODEL_CHOICES[:25],
    "lora": SDXL_LORA_CHOICES[:25],
    "lora2": SDXL_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
CASCADE_ARG_CHOICES = {
    "aspect_ratio": ASPECT_RATIO_CHOICES,
    "lora": CASCADE_LORA_CHOICES[:25],
    "lora2": CASCADE_LORA_CHOICES[:25],
}
SD3_ARG_CHOICES = {
    "model": SD3_MODEL_CHOICES[:25],
    "lora": SD3_LORA_CHOICES[:25],
    "lora2": SD3_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
FLUX_ARG_CHOICES = {
    "model": FLUX_MODEL_CHOICES[:25],
    "lora": FLUX_LORA_CHOICES[:25],
    "lora2": FLUX_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
VIDEO_ARG_CHOICES = {
    "lora": VIDEO_LORA_CHOICES[:25],
}

PONY_ARG_CHOICES = {
    "model": PONY_MODEL_CHOICES[:25],
    "lora": PONY_LORA_CHOICES[:25],
    "lora2": PONY_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
