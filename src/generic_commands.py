from typing import Optional

import discord
from discord import app_commands

from src.image_gen.model_definitions.model_definitions import *

info_commands = []

COMMAND_MODEL_CHOICES = {
    "sdxl": SDXLModelDefinition().model_choices,
    "legacy": SD15ModelDefinition().model_choices,
    "pony": PonyModelDefinition().model_choices,
    "cascade": CascadeModelDefinition().model_choices,
    "sd3": SD3ModelDefinition().model_choices,
    "flux": FluxModelDefinition().model_choices,
    "edit": FluxKontextModelDefinition().model_choices,
}

COMMAND_LORA_CHOICES = {
    "sdxl": SDXLModelDefinition().lora_choices,
    "legacy": SD15ModelDefinition().lora_choices,
    "pony": PonyModelDefinition().lora_choices,
    "cascade": CascadeModelDefinition().lora_choices,
}

class HelpCommands:
    def __init__(self, tree: discord.app_commands.CommandTree):
        self.tree = tree

    def add_commands(self):
        @self.tree.command(name="help", description="Get help")
        async def slash_command(interaction: discord.Interaction):
            await interaction.response.send_message(
                f"""Here are the commands:
`/imagine` - Generate an image using SDXL. Fast
`/sdxl` - Same as the above.
`/cascade` - Use Stable Cascade to generate an image. Does text well. Use the two image inputs to do an image mashup!
`/pony` - Use Pony Diffusion to generate an image. Based on SDXL.
`/legacy` - Use SD 1.5 to generate an image.
`/video` - Create a video with an input image.
`/models` - List all available models.
`/loras` - List all available Loras.
""",
                ephemeral=True,
            )


class InfoCommands:
    def __init__(self, tree: discord.app_commands.CommandTree):
        self.tree = tree

    def add_commands(self):
        @self.tree.command(name="models", description="List all available models")
        @app_commands.describe(command="The command to list models from")
        @app_commands.choices(command=[app_commands.Choice(name=k, value=k) for k in COMMAND_MODEL_CHOICES.keys()])
        async def slash_command(interaction: discord.Interaction, command: Optional[str] = None):
            if command is None:
                model_string = ""
                for mode in COMMAND_MODEL_CHOICES.keys():
                    model_string += f"**/{mode}**:\n```"
                    model_string += "\n".join([m.name for m in COMMAND_MODEL_CHOICES[mode]])
                    model_string += "```\n"
            else:
                model_string = f"**/{command}**\n```"
                model_string += "\n".join([m.name for m in COMMAND_MODEL_CHOICES[command]])
                model_string += "```"

            await interaction.response.send_message(
                f"""Here are the models:
                
{model_string}

            """,
                ephemeral=True,
            )

        @self.tree.command(name="loras", description="List all available loras")
        @app_commands.describe(command="The command to list loras from")
        @app_commands.choices(command=[app_commands.Choice(name=k, value=k) for k in COMMAND_LORA_CHOICES.keys()])
        async def slash_command(interaction: discord.Interaction, command: Optional[str] = None):
            modes = COMMAND_LORA_CHOICES.keys()

            if command is None:
                lora_string = ""
                for mode in modes:
                    lora_string += f"**/{mode}**:\n```"
                    lora_string += "\n".join([m.name for m in COMMAND_LORA_CHOICES[mode]])
                    lora_string += "```\n"
            else:
                lora_string = f"**/{command}**\n```"
                lora_string += "\n".join([m.name for m in COMMAND_LORA_CHOICES[command]])
                lora_string += "```"

            await interaction.response.send_message(
                f"""Here are the loras:
                
{lora_string}

            """,
                ephemeral=True,
            )
