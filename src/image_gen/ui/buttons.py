import asyncio
import os
import random
from copy import deepcopy
from datetime import datetime

import discord
import discord.ext
from discord import SelectOption

from src.comfy_workflows import do_workflow
from src.defaults import *
from src.image_gen.collage_utils import create_collage, create_gif_collage
from src.image_gen.nsfw_detection import check_nsfw
from src.util import get_filename, build_command, get_workflow


# <editor-fold desc="ButtonDecorators">
class EditableButton:
    @discord.ui.button(label="Edit", style=discord.ButtonStyle.blurple, emoji="📝", row=0)
    async def edit_image(self, interaction, button):
        task = asyncio.create_task(self._edit_image(interaction, button))
        await task

    async def _edit_image(self, interaction, button):
        edit_view = EditResponse(self.params, self.command, self.images)
        await edit_view.show_edit_message(interaction)


class RerollableButton:
    @discord.ui.button(label="Re-roll", style=discord.ButtonStyle.green, emoji="🎲", row=0)
    async def reroll_image(self, interaction, btn):
        task = asyncio.create_task(self._reroll_image(interaction, btn))
        await task

    async def _reroll_image(self, interaction, btn):
        await interaction.response.send_message(f'{interaction.user.mention} asked me to re-imagine "{self.params.prompt}", this shouldn\'t take too long...')
        btn.disabled = True
        await interaction.message.edit(view=self)

        params = deepcopy(self.params)

        params.seed = random.randint(0, 999999999999999)

        images = await do_workflow(params, interaction)

        if self.is_video:
            collage_path = create_gif_collage(images)
            fname = "collage.gif"
        else:
            collage_path = create_collage(images, params)
            fname = "collage.png"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(collage_path, params.prompt)

        final_message = f'{interaction.user.mention} asked me to re-imagine "{params.prompt}", here is what I imagined for them. ' f"Seed: {params.seed}"
        buttons = Buttons(params, images, interaction.user, command=self.command)

        await interaction.channel.send(content=final_message, file=discord.File(fp=collage_path, filename=fname, spoiler=is_nsfw), view=buttons)


class DeletableButton:
    def __init__(self, author):
        self.author = author

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.red, emoji="🗑️", row=1)
    async def delete_image_post(self, interaction, button):
        task = asyncio.create_task(self._delete_image_post(interaction, button))
        await task

    async def _delete_image_post(self, interaction, button):
        if interaction.user.id != self.author.id:
            return

        await interaction.message.delete()


class InfoableButton:
    @discord.ui.button(label="Info", style=discord.ButtonStyle.blurple, emoji="ℹ️", row=1)
    async def image_info(self, interaction, button):
        task = asyncio.create_task(self._image_info(interaction, button))
        await task

    async def _image_info(self, interaction, button):
        params = self.params
        info_str = (
            f"prompt: {params.prompt}\n"
            f"negative prompt: {params.negative_prompt}\n"
            f"model: {params.model or 'default'}\n"
            f"loras: {params.loras}\n"
            f"lora strengths: {params.lora_strengths}\n"
            f"aspect ratio: {params.dimensions or 'default'}\n"
            f"sampler: {params.sampler or 'default'}\n"
            f"num steps: {params.num_steps or 'default'}\n"
            f"cfg scale: {params.cfg_scale or 'default'}\n"
            f"seed: {params.seed}\n"
            f"clip skip: {params.clip_skip}\n"
            f"aspect_ratio: {params.dimensions}\n"
            f"```{build_command(params)}```"
        )
        files = []
        if params.filename is not None and params.filename != "" and os.path.exists(params.filename):
            info_str += f"\noriginal file:"
            files.append(discord.File(fp=params.filename, filename=params.filename))

        if params.filename2 is not None and params.filename2 != "" and os.path.exists(params.filename2):
            info_str += f"\noriginal file 2:"
            files.append(discord.File(fp=params.filename2, filename=params.filename2))

        if files:
            await interaction.response.send_message(info_str, files=files, ephemeral=True)
            return

        await interaction.response.send_message(info_str, ephemeral=True)


# </editor-fold>


class ImageButton(discord.ui.Button):
    def __init__(self, label, emoji, row, callback):
        super().__init__(label=label, style=discord.ButtonStyle.grey, emoji=emoji, row=row)
        self._callback = callback

    async def callback(self, interaction: discord.Interaction):
        await self._callback(interaction, self)


class AddDetailDropdown(discord.ui.Select):
    def __init__(self, placeholder, options, row):
        super().__init__(placeholder=placeholder, options=options, row=row)
        self.denoise_strength = ADD_DETAIL_DEFAULTS.denoise_strength

    async def callback(self, interaction: discord.Interaction):
        self.denoise_strength = float(self.values[0])
        await interaction.response.defer()


class Buttons(discord.ui.View, EditableButton, RerollableButton, DeletableButton, InfoableButton):
    def __init__(
            self,
            params,
            images,
            author,
            *,
            timeout=None,
            command=None,
    ):
        if images is None:
            return

        super().__init__(timeout=timeout)
        self.params = params
        self.images = images
        self.author = author
        self.command = command

        self.is_video = command == "video"

        # upscaling/alternative buttons not needed for video
        if self.is_video:
            return

        total_buttons = len(images) * 2 + 1  # For both alternative and upscale buttons + re-roll button
        if total_buttons > 25:  # Limit to 25 buttons
            images = images[:12]  # Adjust to only use the first 12 images

        # Determine if re-roll button should be on its own row
        reroll_row = 2 if total_buttons <= 21 else 0

        # Dynamically add alternative buttons
        for idx, _ in enumerate(images):
            row = (idx + 1) // 5 + reroll_row
            btn = ImageButton(f"V{idx + 1}", "♻️", row, self.generate_alternatives_and_send)
            self.add_item(btn)

        # Dynamically add upscale buttons
        for idx, _ in enumerate(images):
            row = (idx + len(images) + 1) // 5 + reroll_row
            btn = ImageButton(f"U{idx + 1}", "⬆️", row, self.upscale_and_send)
            self.add_item(btn)

        # Dynamically add download buttons
        for idx, _ in enumerate(images):
            row = (idx + (len(images) * 2) + 2) // 5 + reroll_row
            btn = ImageButton(f"D{idx + 1}", "💾", row, self.download_image)
            self.add_item(btn)

    async def generate_alternatives_and_send(self, interaction, button):
        task = asyncio.create_task(self._generate_alternatives_and_send(interaction, button))
        await task

    async def _generate_alternatives_and_send(self, interaction, button):
        index = int(button.label[1:]) - 1  # Extract index from label
        await interaction.response.send_message(
            f"{interaction.user.mention} asked for some alternatives of image #{index + 1}, this shouldn't take too long..."
        )

        params = deepcopy(self.params)
        params.workflow_type = WorkflowType.img2img
        params.seed = random.randint(0, 999999999999999)
        params.denoise_strength = (
            SDXL_GENERATION_DEFAULTS.denoise_strength if self.params.model_type == ModelType.SDXL else SD15_GENERATION_DEFAULTS.denoise_strength
        )

        # TODO: should alternatives use num_steps and cfg_scale from original?
        # Buttons should probably still receive these params for rerolls
        params.filename = os.path.join(os.getcwd(), f"out/images_{get_filename(interaction, self.params)}_{index}.png")
        self.images[index].save(fp=params.filename)
        images = await do_workflow(params, interaction)
        collage_path = create_collage(images, params)
        final_message = f"{interaction.user.mention} here are your alternative images"

        buttons = Buttons(params, images, interaction.user, command=self.command)

        fname = "collage.gif" if images[0].format == "GIF" else "collage.png"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(collage_path, params.prompt)

        await interaction.channel.send(content=final_message, file=discord.File(fp=collage_path, filename=fname, spoiler=is_nsfw), view=buttons)

    async def upscale_and_send(self, interaction, button):
        task = asyncio.create_task(self._upscale_and_send(interaction, button))
        await task

    async def _upscale_and_send(self, interaction, button):
        index = int(button.label[1:]) - 1  # Extract index from label
        await interaction.response.send_message(f"{interaction.user.mention} asked for an upscale of image #{index + 1}, this shouldn't take too long...")

        params = deepcopy(self.params)
        params.workflow_type = WorkflowType.upscale
        params.batch_size = UPSCALE_DEFAULTS.batch_size

        params.filename = os.path.join(os.getcwd(), f"out/images_{get_filename(interaction, self.params)}_{index}.png")
        self.images[index].save(fp=params.filename)
        upscaled_image = await do_workflow(params, interaction)

        if upscaled_image is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        pnginfo = get_workflow(upscaled_image, params)
        upscaled_image_path = f"./out/upscaledImage_{timestamp}.png"
        upscaled_image.save(upscaled_image_path, pnginfo=pnginfo)
        final_message = f"{interaction.user.mention} here is your upscaled image"
        buttons = AddDetailButtons(params, upscaled_image, author=interaction.user)
        fp = f"{get_filename(interaction, self.params)}_{index}.png"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(upscaled_image_path, params.prompt)

        await interaction.channel.send(content=final_message, file=discord.File(fp=upscaled_image_path, filename=fp, spoiler=is_nsfw), view=buttons)

    async def download_image(self, interaction, button):
        task = asyncio.create_task(self._download_image(interaction, button))
        await task

    async def _download_image(self, interaction, button):
        index = int(button.label[1:]) - 1
        file_name = f"{get_filename(interaction, self.params)}_{index}.png"
        pnginfo = get_workflow(self.images[index], self.params)
        fp = f"./out/images_{file_name}"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(fp, self.params.prompt)

        self.images[index].save(fp, pnginfo=pnginfo)
        await interaction.response.send_message(f"{interaction.user.mention}, here is your image!", file=discord.File(fp=fp, filename=file_name, spoiler=is_nsfw))


class AddDetailButtons(discord.ui.View, DeletableButton, InfoableButton):
    def __init__(self, params, images, *, timeout=None, author=None):
        super().__init__(timeout=timeout)
        self.params = params
        self.images = images
        self.author = author

        if self.params.inpainting_prompt is None:
            options = [
                SelectOption(label="Enhance 20%", value=str(0.2)),
                SelectOption(label="Enhance 30%", value=str(0.3)),
                SelectOption(label="Enhance 40%", value=str(0.4)),
                SelectOption(label="Enhance 50%", value=str(0.5)),
                SelectOption(label="Enhance 60%", value=str(0.6)),
                SelectOption(label="Enhance 70%", value=str(0.7)),
                SelectOption(label="Enhance 80%", value=str(0.8)),
                SelectOption(label="Enhance 90%", value=str(0.9)),
                SelectOption(label="Enhance 100%", value=str(1.0)),
            ]
            self.dropdown = AddDetailDropdown(
                "(Optional) Enhancement %",
                options,
                2,
            )
            self.add_item(ImageButton("Enhance", "🔎", 1, self.add_detail))
            self.add_item(self.dropdown)

    async def add_detail(self, interaction, button):
        task = asyncio.create_task(self._add_detail(interaction, self.dropdown.denoise_strength))
        await task

    async def _add_detail(self, interaction, value):
        await interaction.response.send_message("Increasing detail in the image, this shouldn't take too long...")

        params = deepcopy(self.params)
        params.workflow_type = WorkflowType.add_detail
        params.denoise_strength = value
        params.detailing_controlnet_strength = ADD_DETAIL_DEFAULTS.detailing_controlnet_strength
        params.detailing_controlnet_end_percent = ADD_DETAIL_DEFAULTS.detailing_controlnet_end_percent
        params.seed = random.randint(0, 999999999999999)
        params.batch_size = 1

        params.filename = os.path.join(os.getcwd(), f"out/images_{get_filename(interaction, params)}_{params.seed}.png")
        self.images.save(fp=params.filename)
        images = await do_workflow(params, interaction)

        if images is None or (isinstance(images, list) and len(images) == 0):
            return

        collage_path = create_collage(images, params)
        final_message = f"{interaction.user.mention} here is your image with more detail"

        fp = f"{get_filename(interaction, params)}_detail.png"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(fp, params.prompt)

        await interaction.channel.send(
            content=final_message, file=discord.File(fp=collage_path, filename=fp, spoiler=is_nsfw), view=AddDetailButtons(params, images[0], author=interaction.user)
        )


class EditResponse(discord.ui.View):
    def __init__(self, params: ImageWorkflow, command: str, images):
        super().__init__(timeout=None)
        self.params = params
        self.command = command
        self.images = images

    @discord.ui.button(label="Prompts", style=discord.ButtonStyle.blurple, emoji="📝", row=0)
    async def edit_prompts(self, interaction, button):
        class EditPromptModal(discord.ui.Modal, title="Edit Prompts"):
            def __init__(self, params: ImageWorkflow, command: str, owner: EditResponse):
                super().__init__(timeout=None)
                self.params = params
                self.command = command
                self.owner = owner

                self.positive_prompt = discord.ui.TextInput(
                    label="Prompt",
                    placeholder="Enter a prompt",
                    required=True,
                    default=self.params.prompt,
                    style=discord.TextStyle.paragraph
                )

                self.negative_prompt = discord.ui.TextInput(
                    label="Negative Prompt",
                    placeholder="Enter a negative prompt",
                    required=False,
                    default=self.params.negative_prompt or "",
                    style=discord.TextStyle.paragraph
                )

                self.add_item(self.positive_prompt)
                self.add_item(self.negative_prompt)

            async def on_submit(self, interaction):
                params = deepcopy(self.params)
                params.prompt = self.positive_prompt.value
                params.negative_prompt = self.negative_prompt.value

                await self.owner.generate_with_new_params(interaction, params)

        prompt_modal = EditPromptModal(self.params, self.command, self)
        await interaction.response.send_modal(prompt_modal)

    @discord.ui.button(label="Models", style=discord.ButtonStyle.blurple, emoji="🤖", row=0)
    async def edit_models(self, interaction, button):
        class EditModelModal(discord.ui.Modal, title="Edit Models"):
            def __init__(self, params: ImageWorkflow, command: str, owner: EditResponse):
                super().__init__(timeout=None)
                self.params = params
                self.command = command
                self.owner = owner

                self.model_selection = discord.ui.TextInput(
                    label="Model",
                    placeholder="Select a model",
                    required=True,
                    default=self.params.model,
                )

                self.lora_selection_1 = discord.ui.TextInput(
                    label="LoRA 1",
                    placeholder="Select a LoRA",
                    required=False,
                    default=self.params.loras[0].replace(".safetensors", "") if len(self.params.loras) > 0 and self.params.loras[0] else "",
                )

                self.lora_strength_1 = discord.ui.TextInput(
                    label="LoRA 1 Strength",
                    placeholder="Select a LoRA Strength",
                    required=False,
                    default=str(self.params.lora_strengths[0] if len(self.params.loras) > 0 else 1.0),
                )

                self.lora_selection_2 = discord.ui.TextInput(
                    label="LoRA 2",
                    placeholder="Select a LoRA",
                    required=False,
                    default=self.params.loras[1].replace(".safetensors", "") if len(self.params.loras) > 1 and self.params.loras[1] else "",
                )

                self.lora_strength_2 = discord.ui.TextInput(
                    label="LoRA 2 Strength",
                    placeholder="Select a LoRA Strength",
                    required=False,
                    default=str(self.params.lora_strengths[1] if len(self.params.loras) > 1 else 1.0),
                )

                self.add_item(self.model_selection)
                self.add_item(self.lora_selection_1)
                self.add_item(self.lora_strength_1)
                self.add_item(self.lora_selection_2)
                self.add_item(self.lora_strength_2)

            async def on_submit(self, interaction):
                from src.command_descriptions import loras
                params = deepcopy(self.params)
                params.model = self.model_selection.value
                params.loras = []
                params.lora_strengths = []
                if self.lora_selection_1.value:
                    lora = next((lora for lora in loras if self.lora_selection_1.value in lora), self.lora_selection_1.value)
                    params.loras.append(lora)
                    params.lora_strengths.append(float(self.lora_strength_1.value))
                if self.lora_selection_2.value:
                    lora = next((lora for lora in loras if self.lora_selection_2.value in lora), self.lora_selection_2.value)
                    params.loras.append(lora)
                    params.lora_strengths.append(float(self.lora_strength_2.value))

                await self.owner.generate_with_new_params(interaction, params)

        model_modal = EditModelModal(self.params, self.command, self)
        await interaction.response.send_modal(model_modal)

    @discord.ui.button(label="Sampler Parameters", style=discord.ButtonStyle.blurple, emoji="📊", row=0)
    async def edit_sampler_params(self, interaction, button):
        class EditSamplerModal(discord.ui.Modal, title="Edit Sampler Parameters"):
            def __init__(self, params: ImageWorkflow, command: str, owner: EditResponse):
                super().__init__(timeout=None)
                self.params = params
                self.command = command
                self.owner = owner

                self.num_steps = discord.ui.TextInput(
                    label="Num Steps",
                    placeholder="Select number of steps",
                    required=False,
                    default=str(self.params.num_steps),
                )

                self.cfg_scale = discord.ui.TextInput(
                    label="CFG Scale",
                    placeholder="Select CFG scale",
                    required=False,
                    default=str(self.params.cfg_scale),
                )

                self.seed = discord.ui.TextInput(
                    label="Seed",
                    placeholder="Select seed",
                    required=False,
                    default=str(self.params.seed),
                )

                self.sampler = discord.ui.TextInput(
                    label="Sampler",
                    placeholder="Select sampler",
                    required=False,
                    default=str(self.params.sampler),
                )

                self.scheduler = discord.ui.TextInput(
                    label="Scheduler",
                    placeholder="Select scheduler",
                    required=False,
                    default=str(self.params.scheduler or "normal"),
                )

                self.add_item(self.num_steps)
                self.add_item(self.cfg_scale)
                self.add_item(self.seed)
                self.add_item(self.sampler)
                self.add_item(self.scheduler)

            async def on_submit(self, interaction):
                params = deepcopy(self.params)
                params.num_steps = int(self.num_steps.value)
                params.cfg_scale = float(self.cfg_scale.value)
                params.seed = int(self.seed.value)
                params.sampler = self.sampler.value
                params.scheduler = self.scheduler.value

                await self.owner.generate_with_new_params(interaction, params)

        sampler_modal = EditSamplerModal(self.params, self.command, self)
        await interaction.response.send_modal(sampler_modal)

    @discord.ui.button(label="Inpainting", style=discord.ButtonStyle.blurple, emoji="🖌️", row=0)
    async def inpainting_tools(self, interaction, button):
        class EditInpaintingModal(discord.ui.Modal, title="Edit Inpainting"):
            def __init__(self, params: ImageWorkflow, images, owner: EditResponse):
                super().__init__(timeout=None)
                self.params = params
                self.images = images
                self.owner = owner

                self.prompt = discord.ui.TextInput(
                    label="Prompt",
                    placeholder="Enter a prompt",
                    required=True,
                    default=self.params.prompt,
                    style=discord.TextStyle.paragraph
                )

                self.inpainting_prompt = discord.ui.TextInput(
                    label="Inpainting Prompt",
                    placeholder="Enter inpainting prompt",
                    required=True,
                    default=self.params.inpainting_prompt or ""
                )

                self.inpainting_threshold = discord.ui.TextInput(
                    label="Inpainting Detection Threshold",
                    placeholder="Enter inpainting detection threshold",
                    required=True,
                    default=str(self.params.inpainting_detection_threshold)
                )

                self.selection = discord.ui.TextInput(
                    label="Batch Selection",
                    placeholder="Enter batch selection",
                    required=True,
                    default=str(1)
                )

                self.denoising_strength = discord.ui.TextInput(
                    label="Denoising Strength",
                    placeholder="Enter denoising strength",
                    required=False,
                    default=str(0.3)
                )

                self.add_item(self.prompt)
                self.add_item(self.inpainting_prompt)
                self.add_item(self.inpainting_threshold)
                self.add_item(self.selection)
                self.add_item(self.denoising_strength)

            async def on_submit(self, interaction):
                params = deepcopy(self.params)
                params.prompt = self.prompt.value
                params.inpainting_prompt = self.inpainting_prompt.value
                params.inpainting_detection_threshold = float(self.inpainting_threshold.value)
                params.denoise_strength = self.denoising_strength.value
                selection = int(self.selection.value) - 1
                params.filename = os.path.join(os.getcwd(), f"out/images_{get_filename(interaction, self.params)}_{selection}.png")
                self.images[selection].save(fp=params.filename)
                params.workflow_type = WorkflowType.img2img

                await self.owner.generate_with_new_params(interaction, params)

        inpainting_modal = EditInpaintingModal(self.params, self.images, self)
        await interaction.response.send_modal(inpainting_modal)


    async def generate_with_new_params(self, interaction, params):
        await interaction.response.send_message(f"Generating image with new parameters, this shouldn't take too long...")
        images = await do_workflow(params, interaction)
        final_message = f'{interaction.user.mention} asked me to re-imagine "{params.prompt}", here is what I imagined for them. Seed: {params.seed}'
        buttons = Buttons(params, images, interaction.user, command=self.command)
        if self.command == "video": 
            collage_path = create_gif_collage(images)
            fname = "collage.gif"
        else:
            collage_path = create_collage(images, params)
            fname = "collage.png"

        is_nsfw = False
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
            is_nsfw = check_nsfw(collage_path, params.prompt)

        await interaction.channel.send(content=final_message, file=discord.File(fp=collage_path, filename=fname, spoiler=is_nsfw), view=buttons)

    async def show_edit_message(self, interaction):
        await interaction.response.send_message("Here are some options to edit your images!", view=self, ephemeral=True)
