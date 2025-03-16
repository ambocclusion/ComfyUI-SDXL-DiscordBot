import configparser

from src.image_gen.ImageWorkflow import *

config = configparser.ConfigParser()
config.read("config.properties")

def get_default_from_config(section : str, option : str) -> str:
    if section not in config:
        print(f"Section {section} not found in config")
        return None
    if option not in config[section]:
        return None
    return config[section][option]

SD15_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SD15, # model_type
    None, # workflow_type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("SD15_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths TODO add lora and lora strength defaults
    get_default_from_config("SD15_GENERATION_DEFAULTS", "ASPECT_RATIO"),
    get_default_from_config("SD15_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("SD15_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("SD15_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("SD15_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("SD15_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "imagine",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("SD15_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "CLIP_SKIP")),  # clip_skip
    scheduler=get_default_from_config("SD15_GENERATION_DEFAULTS", "SCHEDULER"),
    llm_profile=get_default_from_config("SD15_GENERATION_DEFAULTS", "LLM_PROFILE"),
    use_tensorrt=bool(get_default_from_config("SD15_GENERATION_DEFAULTS", "USE_TENSORRT")) or False,
    tensorrt_model=get_default_from_config("SD15_GENERATION_DEFAULTS", "TENSORRT_MODEL"),
)

SDXL_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SDXL,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "ASPECT_RATIO"),
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("SDXL_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("SDXL_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "sdxl",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "CLIP_SKIP")),  # clip_skip
    None,   # filename2
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "ACCELERATOR_ENABLED"),
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "ACCELERATOR_LORA_NAME"),
    get_default_from_config("SDXL_GENERATION_DEFAULTS", "SCHEDULER"),
    style_prompt=get_default_from_config("SDXL_GENERATION_DEFAULTS", "DEFAULT_STYLE_PROMPT"),
    negative_style_prompt=get_default_from_config("SDXL_GENERATION_DEFAULTS", "DEFAULT_NEGATIVE_STYLE_PROMPT"),
    detailing_controlnet=get_default_from_config("SDXL_GENERATION_DEFAULTS", "DETAILING_CONTROLNET"),
    llm_profile=get_default_from_config("SDXL_GENERATION_DEFAULTS", "LLM_PROFILE"),
    use_align_your_steps=get_default_from_config("SDXL_GENERATION_DEFAULTS", "USE_ALIGN_YOUR_STEPS"),
    use_tensorrt=bool(get_default_from_config("SDXL_GENERATION_DEFAULTS", "USE_TENSORRT")),
    tensorrt_model=get_default_from_config("SDXL_GENERATION_DEFAULTS", "TENSORRT_MODEL"),
)

CASCADE_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.CASCADE,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("CASCADE_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    get_default_from_config("CASCADE_GENERATION_DEFAULTS", "ASPECT_RATIO"),  # aspect_ratio
    get_default_from_config("CASCADE_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("CASCADE_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("CASCADE_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("CASCADE_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("CASCADE_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "cascade",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("SDXL_GENERATION_DEFAULTS", "CLIP_SKIP")),  # clip_skip
    llm_profile=get_default_from_config("CASCADE_GENERATION_DEFAULTS", "LLM_PROFILE"),
)

SVD_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("VIDEO_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    get_default_from_config("VIDEO_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "CFG_SCALE")),
    int(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "svd",  # slash_command
    min_cfg=float(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "MIN_CFG")),
    motion=int(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "MOTION")),
    augmentation=float(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "AUGMENTATION")),
    fps=int(get_default_from_config("VIDEO_GENERATION_DEFAULTS", "FPS")),
)

WAN_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    WorkflowType.wan,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("WAN_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    get_default_from_config("WAN_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("WAN_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("WAN_GENERATION_DEFAULTS", "CFG_SCALE")),
    int(get_default_from_config("WAN_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "video",  # slash_command
    fps=int(get_default_from_config("WAN_GENERATION_DEFAULTS", "FPS")),
    style_prompt=get_default_from_config("WAN_GENERATION_DEFAULTS", "DEFAULT_STYLE_PROMPT"),
    negative_style_prompt=get_default_from_config("WAN_GENERATION_DEFAULTS", "DEFAULT_NEGATIVE_STYLE_PROMPT"),
)

IMAGE_WAN_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    WorkflowType.image_wan,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "CFG_SCALE")),
    int(get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "video",  # slash_command
    fps=int(get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "FPS")),
    style_prompt=get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "DEFAULT_STYLE_PROMPT"),
    negative_style_prompt=get_default_from_config("IMAGE_WAN_GENERATION_DEFAULTS", "DEFAULT_NEGATIVE_STYLE_PROMPT"),
)


PONY_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SDXL,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("PONY_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    get_default_from_config("PONY_GENERATION_DEFAULTS", "ASPECT_RATIO"),  # aspect_ratio
    get_default_from_config("PONY_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("PONY_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("PONY_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("PONY_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("PONY_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "pony",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("PONY_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("PONY_GENERATION_DEFAULTS", "CLIP_SKIP")),
    style_prompt=get_default_from_config("PONY_GENERATION_DEFAULTS", "DEFAULT_STYLE_PROMPT"),
    negative_style_prompt=get_default_from_config("PONY_GENERATION_DEFAULTS", "DEFAULT_NEGATIVE_STYLE_PROMPT"),
    vae=get_default_from_config("PONY_GENERATION_DEFAULTS", "VAE"),
    detailing_controlnet=get_default_from_config("PONY_GENERATION_DEFAULTS", "DETAILING_CONTROLNET"),
    llm_profile=get_default_from_config("PONY_GENERATION_DEFAULTS", "LLM_PROFILE"),
    use_align_your_steps=get_default_from_config("PONY_GENERATION_DEFAULTS", "USE_ALIGN_YOUR_STEPS"),
)

SD3_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SD3,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("SD3_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    get_default_from_config("SD3_GENERATION_DEFAULTS", "ASPECT_RATIO"),  # aspect_ratio
    get_default_from_config("SD3_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("SD3_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("SD3_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("SD3_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("SD3_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "sd3",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("SD3_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("SD3_GENERATION_DEFAULTS", "CLIP_SKIP")),
    llm_profile=get_default_from_config("SD3_GENERATION_DEFAULTS", "LLM_PROFILE"),
    use_align_your_steps=get_default_from_config("SD3_GENERATION_DEFAULTS", "USE_ALIGN_YOUR_STEPS"),
    scheduler=get_default_from_config("SD3_GENERATION_DEFAULTS", "SCHEDULER"),
    use_tensorrt=bool(get_default_from_config("SD3_GENERATION_DEFAULTS", "USE_TENSORRT")) or False,
    tensorrt_model=get_default_from_config("SD3_GENERATION_DEFAULTS", "TENSORRT_MODEL"),
)

FLUX_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.FLUX,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    get_default_from_config("FLUX_GENERATION_DEFAULTS", "MODEL"),
    None,  # loras
    None,  # lora_strengths
    get_default_from_config("FLUX_GENERATION_DEFAULTS", "ASPECT_RATIO"),  # aspect_ratio
    get_default_from_config("FLUX_GENERATION_DEFAULTS", "SAMPLER"),
    int(get_default_from_config("FLUX_GENERATION_DEFAULTS", "NUM_STEPS")),
    float(get_default_from_config("FLUX_GENERATION_DEFAULTS", "CFG_SCALE")),
    float(get_default_from_config("FLUX_GENERATION_DEFAULTS", "DENOISE_STRENGTH")),
    int(get_default_from_config("FLUX_GENERATION_DEFAULTS", "BATCH_SIZE")),  # batch_size
    None,  # seed
    None,  # filename
    "flux",  # slash_command
    None,  # inpainting_prompt
    int(get_default_from_config("FLUX_GENERATION_DEFAULTS", "INPAINTING_DETECTION_THRESHOLD")),  # inpainting_detection_threshold
    int(get_default_from_config("FLUX_GENERATION_DEFAULTS", "CLIP_SKIP")),
    llm_profile=get_default_from_config("FLUX_GENERATION_DEFAULTS", "LLM_PROFILE"),
    use_align_your_steps=get_default_from_config("FLUX_GENERATION_DEFAULTS", "USE_ALIGN_YOUR_STEPS"),
    scheduler=get_default_from_config("FLUX_GENERATION_DEFAULTS", "SCHEDULER"),
    use_tensorrt=bool(get_default_from_config("FLUX_GENERATION_DEFAULTS", "USE_TENSORRT")) or False,
    tensorrt_model=get_default_from_config("FLUX_GENERATION_DEFAULTS", "TENSORRT_MODEL"),
    mashup_image_strength=float(get_default_from_config("FLUX_GENERATION_DEFAULTS", "MASHUP_IMAGE1_STRENGTH")),
    mashup_inputimage_strength=float(get_default_from_config("FLUX_GENERATION_DEFAULTS", "MASHUP_IMAGE2_STRENGTH")),
)

ADD_DETAIL_DEFAULTS = ImageWorkflow(
    None,
    WorkflowType.add_detail,
    None,
    denoise_strength=float(get_default_from_config("ADD_DETAIL_DEFAULTS", "DENOISE_STRENGTH")),
    batch_size=int(get_default_from_config("ADD_DETAIL_DEFAULTS", "BATCH_SIZE")),
    detailing_controlnet_strength=float(get_default_from_config("ADD_DETAIL_DEFAULTS", "DETAILING_CONTROLNET_STRENGTH")),
    detailing_controlnet_end_percent=float(get_default_from_config("ADD_DETAIL_DEFAULTS", "DETAILING_CONTROLNET_END_PERCENT")),
)

UPSCALE_DEFAULTS = ImageWorkflow(
    None,
    WorkflowType.upscale,
    None,
    model=get_default_from_config("UPSCALE_DEFAULTS", "MODEL"),
)

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