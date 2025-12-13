import random
from copy import deepcopy
from io import BytesIO

import discord
from discord import ui

from src.audio_gen.audio_workflow import AudioWorkflowParams, AudioWorkflowType
from src.audio_gen.audio_gen import AudioWorkflow, model_type_to_workflow
from src.image_gen.ui.buttons import ImageButton


class AudioButtons(discord.ui.View):
    def __init__(self, params, sound_fnames, *, timeout=None):
        super().__init__(timeout=timeout)
        self.params = params
        self.sound_fnames = sound_fnames

        workflow = model_type_to_workflow[params.model_type]

        is_remixable = isinstance(workflow, AudioWorkflow.Remixable)
        is_extendable = isinstance(workflow, AudioWorkflow.Extendable)
        if (num_sounds := len(sound_fnames)) > 1:
            row = 1
            if is_remixable:
                for i in range(num_sounds):
                    self.add_item(ImageButton(f"R{i + 1}", "♻️", row, self.remix))
                row += 1
            if is_extendable:
                for i in range(num_sounds):
                    self.add_item(ImageButton(f"E{i + 1}", "⏩", row, self.extend))
        else:
            # just put everything on the first row if there's only one sound
            if is_remixable:
                self.add_item(ImageButton("Remix", "♻️", 0, self.remix))
            if is_extendable:
                self.add_item(ImageButton("Extend", "⏩", 0, self.extend))

    @discord.ui.button(label="Re-roll", style=discord.ButtonStyle.green, emoji="🎲", row=0)
    async def reroll(self, interaction, btn):
        await interaction.response.send_message(
            f'{interaction.user.mention} asked me to re-imagine "{self.params.prompt}", this shouldn\'t take too long...'
        )
        btn.disabled = True
        await interaction.message.edit(view=self)

        params = deepcopy(self.params)
        params.seed = random.randint(0, 999999999999999)

        workflow = model_type_to_workflow[params.model_type](params)
        match params.workflow_type:
            case AudioWorkflowType.GENERATE:
                result = await workflow.generate()
            case AudioWorkflowType.EXTEND:
                result = await workflow.extend()
            case AudioWorkflowType.REMIX:
                result = await workflow.remix()
        videos, _, sound_fnames = result

        files = [discord.File(fp=BytesIO(v), filename=f"sound_{i}.webm") for i, v in enumerate(videos)]
        buttons = AudioButtons(params, sound_fnames)
        await interaction.channel.send(
            content=f"{interaction.user.mention} here is your re-imagined audio",
            files=files,
            view=buttons,
        )

    async def extend(self, interaction, button):
        index = (int(button.label[-1:]) - 1) if button.label != "Extend" else 0
        await interaction.response.send_message(
            f'{interaction.user.mention} asked me to extend "{self.params.prompt}", this shouldn\'t take too long...'
        )

        params: AudioWorkflowParams = deepcopy(self.params)
        params.workflow_type = AudioWorkflowType.EXTEND
        params.duration = None
        params.snd_filename = self.sound_fnames[index]
        params.seed = params.seed or random.randint(0, 999999999999999)

        workflow = model_type_to_workflow[params.model_type](params)
        videos, _, sound_fnames = await workflow.extend()

        files = [discord.File(fp=BytesIO(v), filename=f"sound_{i}.webm") for i, v in enumerate(videos)]
        buttons = AudioButtons(params, sound_fnames)
        await interaction.channel.send(
            content=f"{interaction.user.mention} here is your extended audio",
            files=files,
            view=buttons,
        )

    async def remix(self, interaction, button):
        index = (int(button.label[-1:]) - 1) if button.label != "Remix" else 0

        params: AudioWorkflowParams = deepcopy(self.params)
        params.workflow_type = AudioWorkflowType.REMIX
        params.duration = None
        params.snd_filename = self.sound_fnames[index]
        params.seed = random.randint(0, 999999999999999)

        modal = AudioRemixModal(params)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="Info", style=discord.ButtonStyle.blurple, emoji="ℹ️", row=0)
    async def info(self, interaction, button):
        # TODO: prettify output
        info_str = str(self.params)
        await interaction.response.send_message(info_str, ephemeral=True)


class AudioRemixModal(ui.Modal, title="Remix Sound"):
    def __init__(self, params: AudioWorkflowParams):
        super().__init__(timeout=120)
        self.params = params

        self.prompt = ui.TextInput(
            label="Prompt",
            placeholder="Enter a prompt",
            required=False,
            default=self.params.prompt or "",
            style=discord.TextStyle.paragraph,
        )
        self.lyrics = ui.TextInput(
            label="Lyrics",
            placeholder="Enter lyrics",
            required=False,
            default=self.params.secondary_prompt or "",
            style=discord.TextStyle.paragraph,
        )
        self.remix_strength = ui.TextInput(
            label="Remix strength",
            placeholder="Enter remix strength",
            required=False,
            default="0.5",
        )
        self.add_item(self.remix_strength)
        self.add_item(self.prompt)
        self.add_item(self.lyrics)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message("Generating audio with new parameters, this shouldn't take too long...")

        params = deepcopy(self.params)
        try:
            params.prompt = self.prompt.value
            params.secondary_prompt = self.lyrics.value
            params.denoise_strength = float(self.remix_strength.value or self.remix_strength.default)
        except ValueError:
            interaction.response.send_message(
                "An error occurred while parsing a value you entered. Please check your inputs and try your request again.",
                ephemeral=True,
            )
            return

        workflow = model_type_to_workflow[params.model_type](params)
        videos, _, sound_fnames = await workflow.remix()

        final_message = f"{interaction.user.mention} here is your remixed audio"
        buttons = AudioButtons(params, sound_fnames)

        files = [discord.File(fp=BytesIO(v), filename=f"sound_{i}.webm") for i, v in enumerate(videos)]
        await interaction.channel.send(content=final_message, files=files, view=buttons)
