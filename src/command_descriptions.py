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


VIDEO_LORA_CHOICES = [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, "wan")]
SAMPLER_CHOICES = [Choice(name=s, value=s) for s in samplers if "adaptive" not in s.lower()]
SCHEDULER_CHOICES = [Choice(name=s, value=s) for s in schedulers]

CONTROLNET_CHOICES = [Choice(name="pose", value="pose"), Choice(name="canny", value="canny"), Choice(name="depth", value="depth")]

# COMMAND_MODEL_CHOICES = {
#     "sdxl": SDXLModelDefinition().model_choices,
#     "legacy": SD15ModelDefinition().model_choices,
#     "pony": PonyModelDefinition().model_choices,
#     "cascade": CascadeModelDefinition().model_choices,
#     "sd3": SD3ModelDefinition().model_choices,
#     "flux": FluxModelDefinition().model_choices,
#     "edit": FluxKontextModelDefinition().model_choices,
# }
# 
# COMMAND_LORA_CHOICES = {
#     "sdxl": SDXLModelDefinition().lora_choices,
#     "legacy": SD15ModelDefinition().lora_choices,
#     "pony": PonyModelDefinition().lora_choices,
#     "cascade": CascadeModelDefinition().lora_choices,
# }

BASE_ARG_DESCS = {
    "prompt": "Prompt for the image being generated",
    "negative_prompt": "Prompt for what you want to steer the AI away from",
}

CONTROLNET_ARG_DESCS = {
    "controlnet_type": "Controlnet type to use",
    "controlnet_strength": f"range [0.0, 1.0]; Strength of controlnet",
    "controlnet_start_percent": f"range [0.0, 1.0]; How many steps to wait before applying controlnet",
    "controlnet_end_percent": f"range [0.0, 1.0]; How many steps to wait before stopping controlnet",
}

IMAGE_GEN_DESCS = {
    "model": "Model checkpoint to use",
    "lora": "LoRA to apply",
    "lora_strength": "Strength of LoRA",
    "aspect_ratio": "Aspect ratio of the generated image",
    "num_steps": f"range [1, {MAX_STEPS}]; Number of sampling steps",
    "cfg_scale": f"range [1.0, {MAX_CFG}]; Degree to which AI should follow prompt",
    "input_file": "Image to use as input for img2img",
    "denoise_strength": f"range [0.01, 1.0], default {SD15_GENERATION_DEFAULTS.denoise_strength}; Strength of denoising filter during img2img. Only works when input_file is set",
    "inpainting_prompt": "Detection prompt for inpainting; examples: 'background' or 'person'",
    "inpainting_detection_threshold": f"range [0, 255], default {SD15_GENERATION_DEFAULTS.inpainting_detection_threshold}; Detection threshold for inpainting. Only works when inpainting_prompt is set",
    **CONTROLNET_ARG_DESCS,
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

BASE_ARG_CHOICES = {
    "aspect_ratio": ASPECT_RATIO_CHOICES[:25],
    "controlnet_type": CONTROLNET_CHOICES,
}

VIDEO_ARG_CHOICES = {
    "lora": VIDEO_LORA_CHOICES[:25],
}
