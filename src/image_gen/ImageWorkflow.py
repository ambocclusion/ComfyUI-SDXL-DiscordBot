from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ModelType(Enum):
    SD15 = "SD15",
    SDXL = "SDXL",
    CASCADE = "CASCADE",
    VIDEO = "VIDEO",
    PONY = "PONY",
    SD3 = "SD3",
    FLUX = "FLUX",
    FLUX_KONTEXT = "FLUX_KONTEXT"

class WorkflowType(Enum):
    txt2img = "txt2img",
    img2img = "img2img",
    upscale = "upscale",
    add_detail = "add_detail",
    image_mashup = "image_mashup",
    edit = "edit",
    svd = "svd",
    wan = "wan",
    image_wan = "image_wan"

sd_aspect_ratios = {
    "1:1": (1024, 1024),
    "3:4 portrait": (896, 1152),
    "9:16 portrait": (768, 1344),
    "4:3 landscape": (1152, 896),
    "16:9 landscape": (1344, 768),
}

class ControlnetTypes(Enum):
    pose = "pose",
    canny = "canny",
    depth = "depth",

@dataclass
class ImageWorkflow:
    model_type: ModelType
    workflow_type: WorkflowType

    prompt: str
    negative_prompt: Optional[str] = None

    model: Optional[str] = None
    loras: Optional[list[str]] = None
    lora_strengths: Optional[list[float]] = None

    dimensions: Optional[tuple[int, int]] = None
    sampler: Optional[str] = None
    num_steps: Optional[int] = None
    cfg_scale: Optional[float] = None

    denoise_strength: Optional[float] = None
    batch_size: Optional[int] = None

    seed: Optional[int] = None
    filename: str = None
    slash_command: str = None
    inpainting_prompt: Optional[str] = None
    inpainting_detection_threshold: Optional[float] = None
    clip_skip: Optional[int] = None
    filename2: Optional[str] = None
    use_accelerator_lora: Optional[bool] = None
    accelerator_lora_name: Optional[str] = None
    scheduler: Optional[str] = None
    min_cfg: Optional[float] = None
    motion: Optional[int] = None
    augmentation: Optional[float] = None
    fps: Optional[int] = None
    style_prompt: Optional[str] = None
    negative_style_prompt: Optional[str] = None
    vae: Optional[str] = None
    detailing_controlnet: Optional[str] = None
    detailing_controlnet_strength: Optional[float] = None
    detailing_controlnet_end_percent: Optional[float] = None
    use_llm: Optional[bool] = None
    llm_profile: Optional[str] = None
    use_align_your_steps: Optional[bool] = None
    use_teacache: Optional[bool] = None
    use_triton: Optional[bool] = None
    use_tensorrt: Optional[bool] = None
    tensorrt_model: Optional[str] = None
    mashup_image_strength: Optional[float] = None
    mashup_inputimage_strength: Optional[float] = None
    video_width: Optional[int] = None
    video_length: Optional[int] = None
    controlnet_model: Optional[str] = None
    controlnet_type: Optional[ControlnetTypes] = None
    controlnet_strength: Optional[float] = None
    controlnet_start_percent: Optional[float] = None
    controlnet_end_percent: Optional[float] = None
    clip_model: Optional[str] = None
