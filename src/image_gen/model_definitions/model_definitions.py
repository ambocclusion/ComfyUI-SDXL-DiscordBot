from dataclasses import dataclass

from src.ModelDefinition import ModelDefinition
from src.command_descriptions import BASE_ARG_DESCS, IMAGE_GEN_DESCS
from src.image_gen.ImageWorkflow import ModelType


# SD15_GENERATION_DEFAULTS = get_defaults_for_command("SD15_GENERATION_DEFAULTS", ModelType.SD15, "legacy")
# SDXL_GENERATION_DEFAULTS = get_defaults_for_command("SDXL_GENERATION_DEFAULTS", ModelType.SDXL, "sdxl")
# CASCADE_GENERATION_DEFAULTS = get_defaults_for_command("CASCADE_GENERATION_DEFAULTS", ModelType.CASCADE, "cascade")
# PONY_GENERATION_DEFAULTS = get_defaults_for_command("PONY_GENERATION_DEFAULTS", ModelType.PONY, "pony")
# SD3_GENERATION_DEFAULTS = get_defaults_for_command("SD3_GENERATION_DEFAULTS", ModelType.SD3, "sd3")
# FLUX_GENERATION_DEFAULTS = get_defaults_for_command("FLUX_GENERATION_DEFAULTS", ModelType.FLUX, "flux")
# EDIT_DEFAULTS = get_defaults_for_command("EDIT_DEFAULTS", ModelType.FLUX_KONTEXT, "edit")

@dataclass
class SD15ModelDefinition(ModelDefinition):
    model_name: str = "SD15"
    model_type: ModelType = ModelType.SD15
    slash_command: str = "legacy"
    config_section: str = "SD15_GENERATION"
    model_folder: str = "15"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class SDXLModelDefinition(ModelDefinition):
    model_name: str = "SDXL"
    model_type: ModelType = ModelType.SDXL
    slash_command: str = "sdxl"
    config_section: str = "SDXL_GENERATION"
    model_folder: str = "sdxl"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class CascadeModelDefinition(ModelDefinition):
    model_name: str = "CASCADE"
    model_type: ModelType = ModelType.CASCADE
    slash_command: str = "cascade"
    config_section: str = "CASCADE_GENERATION"
    model_folder: str = "cascade"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class PonyModelDefinition(ModelDefinition):
    model_name: str = "PONY"
    model_type: ModelType = ModelType.PONY
    slash_command: str = "pony"
    config_section: str = "PONY_GENERATION"
    model_folder: str = "pony"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class SD3ModelDefinition(ModelDefinition):
    model_name: str = "SD3"
    model_type: ModelType = ModelType.SD3
    slash_command: str = "sd3"
    config_section: str = "SD3_GENERATION"
    model_folder: str = "sd3"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class FluxModelDefinition(ModelDefinition):
    model_name: str = "FLUX"
    model_type: ModelType = ModelType.FLUX
    slash_command: str = "flux"
    config_section: str = "FLUX_GENERATION"
    model_folder: str = "flux"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }

@dataclass
class FluxKontextModelDefinition(ModelDefinition):
    model_name: str = "FLUX_KONTEXT"
    model_type: ModelType = ModelType.FLUX_KONTEXT
    slash_command: str = "edit"
    config_section: str = "EDIT_GENERATION"
    model_folder: str = "flux_kontext"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
# SVD_GENERATION_DEFAULTS = get_defaults_for_command("SVD_GENERATION_DEFAULTS", ModelType.VIDEO, "video")
# WAN_GENERATION_DEFAULTS = get_defaults_for_command("WAN_GENERATION_DEFAULTS", ModelType.VIDEO, "wan")
# IMAGE_WAN_GENERATION_DEFAULTS = get_defaults_for_command("IMAGE_WAN_GENERATION_DEFAULTS", ModelType.VIDEO, "image_wan")
@dataclass
class SVDModelDefinition(ModelDefinition):
    model_name: str = "VIDEO"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "video"
    config_section: str = "SVD_GENERATION"
    model_folder: str = "video"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class WANModelDefinition(ModelDefinition):
    model_name: str = "WAN"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "wan"
    config_section: str = "WAN_GENERATION"
    model_folder: str = "wan"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class ImageWANModelDefinition(ModelDefinition):
    model_name: str = "IMAGE_WAN"
    model_type: ModelType = ModelType.VIDEO
    slash_command: str = "image_wan"
    config_section: str = "IMAGE_WAN_GENERATION"
    model_folder: str = "image_wan"
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
# ADD_DETAIL_DEFAULTS = get_defaults_for_command("ADD_DETAIL_DEFAULTS", None, "add_detail")
# UPSCALE_DEFAULTS = get_defaults_for_command("UPSCALE_DEFAULTS", None, "upscale")
@dataclass
class AddDetailModelDefinition(ModelDefinition):
    model_name: str = "ADD_DETAIL"
    model_type: ModelType = None
    slash_command: str = "add_detail"
    config_section: str = "ADD_DETAIL"
    model_folder: str = None
    argument_descriptions = {
        **BASE_ARG_DESCS,
        **IMAGE_GEN_DESCS,
    }
    
@dataclass
class UpscaleModelDefinition(ModelDefinition):
    model_name: str = "UPSCALE"
    model_type: ModelType = None
    slash_command: str = "upscale"
    config_section: str = "UPSCALE"