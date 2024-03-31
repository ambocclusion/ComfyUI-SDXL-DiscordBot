import json

from discord.app_commands import Choice

from src.comfyscript_utils import get_models, get_loras, get_samplers
from src.consts import *
from src.defaults import *

models = get_models()
loras = get_loras()
samplers = get_samplers()

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
SAMPLER_CHOICES = [Choice(name=s, value=s) for s in samplers]

COMMAND_MODEL_CHOICES = {
    "sdxl": SDXL_MODEL_CHOICES,
    "legacy": SD15_MODEL_CHOICES,
    "pony": PONY_MODEL_CHOICES,
}

COMMAND_LORA_CHOICES = {
    "sdxl": SDXL_LORA_CHOICES,
    "legacy": SD15_LORA_CHOICES,
    "pony": PONY_LORA_CHOICES,
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
VIDEO_ARG_DESCS = {
    "input_file": "Starting image for video generation",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should adhere to the starting image. Default: {VIDEO_GENERATION_DEFAULTS.cfg_scale}",
    "min_cfg": f"Starting CFG value. Generation will move to CFG_SCALE over the length of the video. Default: {VIDEO_GENERATION_DEFAULTS.min_cfg}",
    "motion": f"The amount of motion in the video. Default: {VIDEO_GENERATION_DEFAULTS.motion}",
    "augmentation": f"How much the video will differ from your starting image. Introduces a lot of noise. Default: {VIDEO_GENERATION_DEFAULTS.augmentation}",
}

CASCADE_ARG_DESCS = {
    **BASE_ARG_DESCS,
    "aspect_ratio": "Aspect ratio of the generated image",
    "num_steps": f"range [1, {MAX_STEPS}]; Number of sampling steps",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should follow prompt",
    "input_file": "Image to use as input for img2img",
    "input_file2": "Image to use for mashup, must have input_file set too",
    "denoise_strength": f"range [0.01, 1.0], default {CASCADE_GENERATION_DEFAULTS.denoise_strength}; Strength of denoising filter during img2img. Only works when input_file is set",
    "inpainting_prompt": "Detection prompt for inpainting; examples: 'background' or 'person'",
    "inpainting_detection_threshold": f"range [0, 255], default {CASCADE_GENERATION_DEFAULTS.inpainting_detection_threshold}; Detection threshold for inpainting. Only works when inpainting_prompt is set",
    "clip_skip": f"default: {CASCADE_GENERATION_DEFAULTS.clip_skip}",
}

PONY_ARG_DESCS = {**SDXL_ARG_DESCS}

BASE_ARG_CHOICES = {
    "aspect_ratio": ASPECT_RATIO_CHOICES,
    "sampler": SAMPLER_CHOICES,
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
}
VIDEO_ARG_CHOICES = {k: v for k, v in IMAGINE_ARG_CHOICES.items() if k not in {"lora2", "lora3", "aspect_ratio"}}

PONY_ARG_CHOICES = {
    "model": PONY_MODEL_CHOICES[:25],
    "lora": PONY_LORA_CHOICES[:25],
    "lora2": PONY_LORA_CHOICES[:25],
    **BASE_ARG_CHOICES,
}
