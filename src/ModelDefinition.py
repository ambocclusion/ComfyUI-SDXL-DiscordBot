from dataclasses import dataclass

from discord.app_commands import Choice

from src.comfyscript_utils import get_models, get_loras
from src.defaults import get_defaults_for_command
from src.image_gen.ImageWorkflow import ModelType, ImageWorkflow


@dataclass
class ModelDefinition:
    model_name: str
    model_type: ModelType
    slash_command: str
    config_section: str
    model_folder: str
    model_choices: list[str]
    lora_choices: list[str]
    argument_descriptions: dict[str, str]
    argument_choices: dict[str, list[any]]
    
    default_image_workflow: ImageWorkflow


    def __init__(self):
        self.model_choices = get_model_choices(self.model_folder)
        self.lora_choices = get_lora_choices(self.model_folder)
        self.default_image_workflow = get_defaults_for_command(self.model_name, self.model_type, self.slash_command)
        

models = get_models()
loras = get_loras()


def get_model_choices(directory: str) -> list[str]:
    return [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, directory)]


def get_lora_choices(directory: str) -> list[str]:
    return [Choice(name=l.replace(".safetensors", ""), value=l) for l in loras if not should_filter_model(l, directory)]

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
