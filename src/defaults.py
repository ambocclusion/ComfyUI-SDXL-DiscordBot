import configparser

from src.image_gen.ImageWorkflow import *

config = configparser.ConfigParser()
config.read("config.properties", encoding="utf8")

def get_default_from_config(section : str, option : str, default = None) -> str:
    if section not in config:
        print(f"Section {section} not found in config")
        return default
    if option not in config[section]:
        return default
    return config[section][option]

def get_defaults_for_command(section: str, model_type: ModelType, slash_command: str) -> ImageWorkflow:
    workflow = ImageWorkflow(
        model_type,
        None, # workflow_type
        None, # prompt
        None, # negative_prompt
        get_default_from_config(section, "MODEL"),
        None, # loras
        None, # lora_strengths TODO add lora and lora strength defaults
        get_default_from_config(section, "ASPECT_RATIO"),
        get_default_from_config(section, "SAMPLER"),
        int(get_default_from_config(section, "NUM_STEPS", 20)),
        float(get_default_from_config(section, "CFG_SCALE", 4.0)),
        float(get_default_from_config(section, "DENOISE_STRENGTH", 0.8)),
        int(get_default_from_config(section, "BATCH_SIZE", 1)), # batch_size
        None, # seed
        None, # filename
        slash_command,
        None, # inpainting_prompt
        int(get_default_from_config(section, "INPAINTING_DETECTION_THRESHOLD", 200)),
        int(get_default_from_config(section, "CLIP_SKIP", -1)),
        None, # filename2
        bool(get_default_from_config(section, "ACCELERATOR_ENABLED", False)),
        get_default_from_config(section, "ACCELERATOR_LORA_NAME"),
        get_default_from_config(section, "SCHEDULER"),
        float(get_default_from_config(section, "MIN_CFG", 0.0)),
        float(get_default_from_config(section, "MOTION", 0.0)),
        float(get_default_from_config(section, "AUGMENTATION", 0.0)),
        int(get_default_from_config(section, "FPS", 0)),
        get_default_from_config(section, "DEFAULT_STYLE_PROMPT"),
        get_default_from_config(section, "DEFAULT_NEGATIVE_STYLE_PROMPT"),
        get_default_from_config(section, "VAE"),
        get_default_from_config(section, "DETAILING_CONTROLNET"),
        get_default_from_config(section, "DETAILING_CONTROLNET_STRENGTH"),
        get_default_from_config(section, "DETAILING_CONTROLNET_END_PERCENT"),
        bool(get_default_from_config(section, "USE_LLM_BY_DEFAULT", False)),
        get_default_from_config(section, "LLM_PROFILE"),
        bool(get_default_from_config(section, "USE_ALIGN_YOUR_STEPS", False)),
        bool(get_default_from_config(section, "USE_TENSORRT", False)),
        get_default_from_config(section, "TENSORRT_MODEL"),
        float(get_default_from_config(section, "MASHUP_IMAGE1_STRENGTH", 1.0)),
        float(get_default_from_config(section, "MASHUP_IMAGE2_STRENGTH", 1.0)) ,
    )
    return workflow

SD15_GENERATION_DEFAULTS = get_defaults_for_command("SD15_GENERATION_DEFAULTS", ModelType.SD15, "imagine")
SDXL_GENERATION_DEFAULTS = get_defaults_for_command("SDXL_GENERATION_DEFAULTS", ModelType.SDXL, "sdxl")
CASCADE_GENERATION_DEFAULTS = get_defaults_for_command("CASCADE_GENERATION_DEFAULTS", ModelType.CASCADE, "cascade")
SVD_GENERATION_DEFAULTS = get_defaults_for_command("SVD_GENERATION_DEFAULTS", ModelType.VIDEO, "video")
WAN_GENERATION_DEFAULTS = get_defaults_for_command("WAN_GENERATION_DEFAULTS", ModelType.VIDEO, "wan")
IMAGE_WAN_GENERATION_DEFAULTS = get_defaults_for_command("IMAGE_WAN_GENERATION_DEFAULTS", ModelType.VIDEO, "image_wan")
PONY_GENERATION_DEFAULTS = get_defaults_for_command("PONY_GENERATION_DEFAULTS", ModelType.PONY, "pony")
SD3_GENERATION_DEFAULTS = get_defaults_for_command("SD3_GENERATION_DEFAULTS", ModelType.SD3, "sd3")
FLUX_GENERATION_DEFAULTS = get_defaults_for_command("FLUX_GENERATION_DEFAULTS", ModelType.FLUX, "flux")
ADD_DETAIL_DEFAULTS = get_defaults_for_command("ADD_DETAIL_DEFAULTS", None, "add_detail")
UPSCALE_DEFAULTS = get_defaults_for_command("UPSCALE_DEFAULTS", None, "upscale")

COMMAND_DEFAULTS = {
    "imagine": SDXL_GENERATION_DEFAULTS,
    "sdxl": SDXL_GENERATION_DEFAULTS,
    "cascade": CASCADE_GENERATION_DEFAULTS,
    "pony": PONY_GENERATION_DEFAULTS,
    "video": SVD_GENERATION_DEFAULTS,
    "add_detail": ADD_DETAIL_DEFAULTS,
    "upscale": UPSCALE_DEFAULTS,
    "sd3": SD3_GENERATION_DEFAULTS,
    "flux": FLUX_GENERATION_DEFAULTS,
}

MAX_RETRIES = int(get_default_from_config("BOT", "MAX_RETRIES") or 3)

llm_prompt = get_default_from_config("LLM", "SYSTEM_PROMPT")

llm_parameters = {
    "API_URL": get_default_from_config("LLM", "API_URL"),
    "API_PORT": get_default_from_config("LLM", "API_PORT"),
    "MODEL_NAME": get_default_from_config("LLM", "MODEL_NAME"),
}