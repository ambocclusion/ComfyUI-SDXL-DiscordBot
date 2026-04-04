from dataclasses import dataclass

from discord.app_commands import Choice

from src.comfyscript_utils import get_models, get_loras
from src.defaults import get_defaults_for_command
from src.image_gen.ImageWorkflow import ModelType
from src.image_gen.generation_workflows.image_workflows import SDWorkflow


@dataclass
class ModelDefinition:
    model_name: str
    model_type: ModelType
    slash_command: str
    config_section: str
    model_folder: str
    workflow: type[SDWorkflow]

    def __init__(self):
        self.model_choices = get_model_choices(self.model_folder)
        self.lora_choices = get_lora_choices(self.model_folder)
        self.default_image_workflow = get_defaults_for_command(f"{self.config_section}_DEFAULTS", self.model_type, self.slash_command)

    def get_general_argument_choices(self):
        return {"model": self.model_choices, "lora": self.lora_choices, "lora2": self.lora_choices}


models = get_models()
loras = get_loras()


def get_model_choices(directory: str) -> list[str]:
    if directory is None:
        return []
    return [Choice(name=m.replace(".safetensors", ""), value=m) for m in models if not should_filter_model(m, directory)]


def get_lora_choices(directory: str) -> list[str]:
    if directory is None:
        return []
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
