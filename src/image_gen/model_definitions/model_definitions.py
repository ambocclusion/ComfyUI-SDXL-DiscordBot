from src.ModelDefinition import ModelDefinition
from src.command_descriptions import BASE_ARG_DESCS, IMAGE_GEN_DESCS, BASE_ARG_CHOICES
from src.image_gen.generation_workflows.image_workflows import *
from src.image_gen.generation_workflows.video_workflows import *


# Image generation model definitions
@dataclass
class SD15ModelDefinition(ModelDefinition):
    model_name: str = "SD15"
    model_type: ModelType = ModelType.SD15
    slash_command: str = "legacy"
    config_section: str = "SD15_GENERATION"
    model_folder: str = "15"
    workflow: type[SDWorkflow] = SD15Workflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class SDXLModelDefinition(ModelDefinition):
    model_name: str = "SDXL"
    model_type: ModelType = ModelType.SDXL
    slash_command: str = "sdxl"
    config_section: str = "SDXL_GENERATION"
    model_folder: str = "sdxl"
    workflow: type[SDWorkflow] = SDXLWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class CascadeModelDefinition(ModelDefinition):
    model_name: str = "CASCADE"
    model_type: ModelType = ModelType.CASCADE
    slash_command: str = "cascade"
    config_section: str = "CASCADE_GENERATION"
    model_folder: str = "cascade"
    workflow: type[SDWorkflow] = SDCascadeWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class PonyModelDefinition(ModelDefinition):
    model_name: str = "PONY"
    model_type: ModelType = ModelType.PONY
    slash_command: str = "pony"
    config_section: str = "PONY_GENERATION"
    model_folder: str = "pony"
    workflow: type[SDWorkflow] = PonyWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class SD3ModelDefinition(ModelDefinition):
    model_name: str = "SD3"
    model_type: ModelType = ModelType.SD3
    slash_command: str = "sd3"
    config_section: str = "SD3_GENERATION"
    model_folder: str = "sd3"
    workflow: type[SDWorkflow] = SD3Workflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class FluxModelDefinition(ModelDefinition):
    model_name: str = "FLUX"
    model_type: ModelType = ModelType.FLUX
    slash_command: str = "flux"
    config_section: str = "FLUX_GENERATION"
    model_folder: str = "flux"
    workflow: type[SDWorkflow] = FluxWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class FluxKontextModelDefinition(ModelDefinition):
    model_name: str = "FLUX_KONTEXT"
    model_type: ModelType = ModelType.FLUX_KONTEXT
    slash_command: str = "edit"
    config_section: str = "EDIT"
    model_folder: str = "flux_kontext"
    workflow: type[SDWorkflow] = FluxWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


# Video generation definitions
@dataclass
class SVDModelDefinition(ModelDefinition):
    model_name: str = "VIDEO"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "video"
    config_section: str = "SVD_GENERATION"
    model_folder: str = "video"
    workflow: type[SDWorkflow] = SVDWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class WANModelDefinition(ModelDefinition):
    model_name: str = "WAN"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "wan"
    config_section: str = "WAN_GENERATION"
    model_folder: str = "wan"
    workflow: type[SDWorkflow] = WANWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


@dataclass
class ImageWANModelDefinition(ModelDefinition):
    model_name: str = "IMAGE_WAN"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "image_wan"
    config_section: str = "IMAGE_WAN_GENERATION"
    model_folder: str = "image_wan"
    workflow: type[SDWorkflow] = WANWorkflow

    def __init__(self):
        super().__init__()
        self.argument_descriptions = {
            **BASE_ARG_DESCS,
            **IMAGE_GEN_DESCS,
        }
        self.argument_choices = {
            **BASE_ARG_CHOICES,
        }


# Misc model definitions
@dataclass
class AddDetailModelDefinition(ModelDefinition):
    model_name: str = "ADD_DETAIL"
    model_type: ModelType = None
    slash_command: str = "add_detail"
    config_section: str = "ADD_DETAIL"
    model_folder: str = None
    workflow: type[SDWorkflow] = None


@dataclass
class UpscaleModelDefinition(ModelDefinition):
    model_name: str = "UPSCALE"
    model_type: ModelType = None
    slash_command: str = "upscale"
    config_section: str = "UPSCALE"
    model_folder: str = None
    workflow: type[SDWorkflow] = UpscaleWorkflow

    def __init__(self):
        super().__init__()
