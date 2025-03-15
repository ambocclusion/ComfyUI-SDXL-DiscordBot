import configparser

from src.image_gen.ImageWorkflow import *

config = configparser.ConfigParser()
config.read("config.properties")

SD15_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SD15, # model_type
    None, # workflow_type
    None,  # prompt
    None,  # negative_prompt
    config["SD15_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths TODO add lora and lora strength defaults
    config["SD15_GENERATION_DEFAULTS"]["ASPECT_RATIO"],
    config["SD15_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["SD15_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["SD15_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["SD15_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["SD15_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "imagine",  # slash_command
    None,  # inpainting_prompt
    int(config["SD15_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["SDXL_GENERATION_DEFAULTS"]["CLIP_SKIP"]),  # clip_skip
    scheduler=config["SD15_GENERATION_DEFAULTS"]["SCHEDULER"] if "SCHEDULER" in config["SD15_GENERATION_DEFAULTS"] else None,
    llm_profile=config["SD15_GENERATION_DEFAULTS"]["LLM_PROFILE"],
    use_tensorrt=bool(config["SD15_GENERATION_DEFAULTS"]["USE_TENSORRT"]) or False,
    tensorrt_model=config["SD15_GENERATION_DEFAULTS"]["TENSORRT_MODEL"],
)

SDXL_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SDXL,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["SDXL_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    config["SDXL_GENERATION_DEFAULTS"]["ASPECT_RATIO"],
    config["SDXL_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["SDXL_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["SDXL_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["SDXL_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["SDXL_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "sdxl",  # slash_command
    None,  # inpainting_prompt
    int(config["SDXL_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["SDXL_GENERATION_DEFAULTS"]["CLIP_SKIP"]),  # clip_skip
    None,   # filename2
    config["SDXL_GENERATION_DEFAULTS"]["ACCELERATOR_ENABLED"],
    config["SDXL_GENERATION_DEFAULTS"]["ACCELERATOR_LORA_NAME"],
    config["SDXL_GENERATION_DEFAULTS"]["SCHEDULER"],
    style_prompt=config["SDXL_GENERATION_DEFAULTS"]["DEFAULT_STYLE_PROMPT"],
    negative_style_prompt=config["SDXL_GENERATION_DEFAULTS"]["DEFAULT_NEGATIVE_STYLE_PROMPT"],
    detailing_controlnet=config["SDXL_GENERATION_DEFAULTS"]["DETAILING_CONTROLNET"],
    llm_profile=config["SDXL_GENERATION_DEFAULTS"]["LLM_PROFILE"],
    use_align_your_steps=config["SDXL_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"],
    use_tensorrt=bool(config["SDXL_GENERATION_DEFAULTS"]["USE_TENSORRT"]),
    tensorrt_model=config["SDXL_GENERATION_DEFAULTS"]["TENSORRT_MODEL"],
)

CASCADE_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.CASCADE,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["CASCADE_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    config["CASCADE_GENERATION_DEFAULTS"]["ASPECT_RATIO"],  # aspect_ratio
    config["CASCADE_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["CASCADE_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["CASCADE_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["CASCADE_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["CASCADE_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "cascade",  # slash_command
    None,  # inpainting_prompt
    int(config["SDXL_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["SDXL_GENERATION_DEFAULTS"]["CLIP_SKIP"]),  # clip_skip
    llm_profile=config["CASCADE_GENERATION_DEFAULTS"]["LLM_PROFILE"],
)

SVD_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["VIDEO_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    config["VIDEO_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["VIDEO_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["VIDEO_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    int(config["VIDEO_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "svd",  # slash_command
    min_cfg=float(config["VIDEO_GENERATION_DEFAULTS"]["MIN_CFG"]),
    motion=int(config["VIDEO_GENERATION_DEFAULTS"]["MOTION"]),
    augmentation=float(config["VIDEO_GENERATION_DEFAULTS"]["AUGMENTATION"]),
    fps=int(config["VIDEO_GENERATION_DEFAULTS"]["FPS"]),
)

WAN_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    WorkflowType.wan,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["WAN_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    config["WAN_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["WAN_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["WAN_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    int(config["WAN_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "video",  # slash_command
    fps=int(config["WAN_GENERATION_DEFAULTS"]["FPS"]),
    style_prompt=config["WAN_GENERATION_DEFAULTS"]["DEFAULT_STYLE_PROMPT"],
    negative_style_prompt=config["WAN_GENERATION_DEFAULTS"]["DEFAULT_NEGATIVE_STYLE_PROMPT"],
)

IMAGE_WAN_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.VIDEO,  # model_type
    WorkflowType.image_wan,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["IMAGE_WAN_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    None,  # aspect_ratio
    config["IMAGE_WAN_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["IMAGE_WAN_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["IMAGE_WAN_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    int(config["IMAGE_WAN_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # denoise_strength
    None,  # seed
    None,  # filename
    "video",  # slash_command
    fps=int(config["IMAGE_WAN_GENERATION_DEFAULTS"]["FPS"]),
    style_prompt=config["IMAGE_WAN_GENERATION_DEFAULTS"]["DEFAULT_STYLE_PROMPT"],
    negative_style_prompt=config["IMAGE_WAN_GENERATION_DEFAULTS"]["DEFAULT_NEGATIVE_STYLE_PROMPT"],
)


PONY_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SDXL,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["PONY_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    config["PONY_GENERATION_DEFAULTS"]["ASPECT_RATIO"],  # aspect_ratio
    config["PONY_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["PONY_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["PONY_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["PONY_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["PONY_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "pony",  # slash_command
    None,  # inpainting_prompt
    int(config["PONY_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["PONY_GENERATION_DEFAULTS"]["CLIP_SKIP"]),
    style_prompt=config["PONY_GENERATION_DEFAULTS"]["DEFAULT_STYLE_PROMPT"],
    negative_style_prompt=config["PONY_GENERATION_DEFAULTS"]["DEFAULT_NEGATIVE_STYLE_PROMPT"],
    vae=config["PONY_GENERATION_DEFAULTS"]["VAE"],
    detailing_controlnet=config["PONY_GENERATION_DEFAULTS"]["DETAILING_CONTROLNET"],
    llm_profile=config["PONY_GENERATION_DEFAULTS"]["LLM_PROFILE"],
    use_align_your_steps=config["PONY_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"],
)

SD3_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.SD3,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["SD3_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    config["SD3_GENERATION_DEFAULTS"]["ASPECT_RATIO"],  # aspect_ratio
    config["SD3_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["SD3_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["SD3_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["SD3_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["SD3_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "sd3",  # slash_command
    None,  # inpainting_prompt
    int(config["SD3_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["SD3_GENERATION_DEFAULTS"]["CLIP_SKIP"]),
    llm_profile=config["SD3_GENERATION_DEFAULTS"]["LLM_PROFILE"],
    use_align_your_steps=config["SD3_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"],
    scheduler=config["SD3_GENERATION_DEFAULTS"]["SCHEDULER"],
    use_tensorrt=bool(config["SD3_GENERATION_DEFAULTS"]["USE_TENSORRT"]) or False,
    tensorrt_model=config["SD3_GENERATION_DEFAULTS"]["TENSORRT_MODEL"],
)

FLUX_GENERATION_DEFAULTS = ImageWorkflow(
    ModelType.FLUX,  # model_type
    None,  # workflow type
    None,  # prompt
    None,  # negative_prompt
    config["FLUX_GENERATION_DEFAULTS"]["MODEL"],
    None,  # loras
    None,  # lora_strengths
    config["FLUX_GENERATION_DEFAULTS"]["ASPECT_RATIO"],  # aspect_ratio
    config["FLUX_GENERATION_DEFAULTS"]["SAMPLER"],
    int(config["FLUX_GENERATION_DEFAULTS"]["NUM_STEPS"]),
    float(config["FLUX_GENERATION_DEFAULTS"]["CFG_SCALE"]),
    float(config["FLUX_GENERATION_DEFAULTS"]["DENOISE_STRENGTH"]),
    int(config["FLUX_GENERATION_DEFAULTS"]["BATCH_SIZE"]),  # batch_size
    None,  # seed
    None,  # filename
    "flux",  # slash_command
    None,  # inpainting_prompt
    int(config["FLUX_GENERATION_DEFAULTS"]["INPAINTING_DETECTION_THRESHOLD"]),  # inpainting_detection_threshold
    int(config["FLUX_GENERATION_DEFAULTS"]["CLIP_SKIP"]),
    llm_profile=config["FLUX_GENERATION_DEFAULTS"]["LLM_PROFILE"],
    use_align_your_steps=config["FLUX_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"],
    scheduler=config["FLUX_GENERATION_DEFAULTS"]["SCHEDULER"],
    use_tensorrt=bool(config["FLUX_GENERATION_DEFAULTS"]["USE_TENSORRT"]) or False,
    tensorrt_model=config["FLUX_GENERATION_DEFAULTS"]["TENSORRT_MODEL"],
    mashup_image_strength=float(config["FLUX_GENERATION_DEFAULTS"]["MASHUP_IMAGE1_STRENGTH"]),
    mashup_inputimage_strength=float(config["FLUX_GENERATION_DEFAULTS"]["MASHUP_IMAGE2_STRENGTH"]),
)

ADD_DETAIL_DEFAULTS = ImageWorkflow(
    None,
    WorkflowType.add_detail,
    None,
    denoise_strength=float(config["ADD_DETAIL_DEFAULTS"]["DENOISE_STRENGTH"]),
    batch_size=int(config["ADD_DETAIL_DEFAULTS"]["BATCH_SIZE"]),
    detailing_controlnet_strength=float(config["ADD_DETAIL_DEFAULTS"]["DETAILING_CONTROLNET_STRENGTH"]),
    detailing_controlnet_end_percent=float(config["ADD_DETAIL_DEFAULTS"]["DETAILING_CONTROLNET_END_PERCENT"]),
)

UPSCALE_DEFAULTS = ImageWorkflow(
    None,
    WorkflowType.upscale,
    None,
    model=config["UPSCALE_DEFAULTS"]["MODEL"],
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

MAX_RETRIES = int(config["BOT"]["MAX_RETRIES"] or 3)

llm_prompt = config["LLM"]["SYSTEM_PROMPT"]

llm_parameters = {
    "API_URL": config["LLM"]["API_URL"],
    "API_PORT": config["LLM"]["API_PORT"],
    "MODEL_NAME": config["LLM"]["MODEL_NAME"],
}