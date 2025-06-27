import logging
import random

import discord
from discord import app_commands, Attachment
from discord.app_commands import Range

from src.command_descriptions import *
from src.consts import *
from src.image_gen.collage_utils import create_collage
from src.image_gen.nsfw_detection import check_nsfw
from src.image_gen.ui.buttons import Buttons
from src.util import process_attachment, unpack_choices, should_filter, get_filename

logger = logging.getLogger("bot")


class ImageGenCommands:
    def __init__(self, tree: discord.app_commands.CommandTree):
        self.tree = tree

    def add_commands(self):

        @self.tree.command(name="svd", description="Generate a video based on an input image using StableVideoDiffusion")
        @app_commands.describe(**SVD_ARG_DESCS)
        # @app_commands.choices(**VIDEO_ARG_CHOICES)
        async def slash_command(
                interaction: discord.Interaction,
                input_file: Attachment,
                cfg_scale: Range[float, 1.0, MAX_CFG] = None,
                min_cfg: Range[float, 1.0, MAX_CFG] = None,
                motion: Range[int, 1, 127] = None,
                augmentation: Range[float, 0, 10] = None,
                seed: int = None,
        ):
            if input_file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                await interaction.response.send_message(
                    f"{interaction.user.mention} `Only PNG, JPG, and JPEG images are supported for video generation`",
                    ephemeral=True,
                )
                return

            params = ImageWorkflow(
                ModelType.VIDEO,
                WorkflowType.svd,
                "",
                "",
                model=SVD_GENERATION_DEFAULTS.model,
                num_steps=SVD_GENERATION_DEFAULTS.num_steps,
                cfg_scale=cfg_scale or SVD_GENERATION_DEFAULTS.cfg_scale,
                seed=seed,
                slash_command="svd",
                sampler=SVD_GENERATION_DEFAULTS.sampler,
                scheduler=SVD_GENERATION_DEFAULTS.scheduler,
                min_cfg=min_cfg or SVD_GENERATION_DEFAULTS.min_cfg,
                motion=motion or SVD_GENERATION_DEFAULTS.motion,
                augmentation=augmentation or SVD_GENERATION_DEFAULTS.augmentation,
                fps=SVD_GENERATION_DEFAULTS.fps,
                filename=await process_attachment(input_file, interaction),
            )
            await self._do_request(
                interaction,
                f"🎥{interaction.user.mention} asked me to create a video with SVD! {random.choice(generation_messages)} 🎥",
                f"{interaction.user.mention} asked me to create video with SVD! {random.choice(completion_messages)} 🎥",
                "video",
                params,
            )

        @self.tree.command(name="video", description="Generate a video based on a prompt")
        @app_commands.describe(**VIDEO_ARG_DESCS)
        @app_commands.choices(**VIDEO_ARG_CHOICES)
        async def slash_command(
                interaction: discord.Interaction,
                prompt: str,
                negative_prompt: str = None,
                cfg_scale: Range[float, 1.0, MAX_CFG] = None,
                input_file: Attachment = None,
                seed: int = None,
                lora: Choice[str] = None,
        ):
            if input_file is not None and input_file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                await interaction.response.send_message(
                    f"{interaction.user.mention} `Only PNG, JPG, and JPEG images are supported for video generation`",
                    ephemeral=True,
                )
                return

            generation_defaults = WAN_GENERATION_DEFAULTS if input_file is None else IMAGE_WAN_GENERATION_DEFAULTS

            params = ImageWorkflow(
                ModelType.VIDEO,
                WorkflowType.wan if input_file == None else WorkflowType.image_wan,
                prompt,
                negative_prompt,
                generation_defaults.model,
                unpack_choices(lora, None),
                [1.0, 1.0],
                num_steps=generation_defaults.num_steps,
                cfg_scale=cfg_scale or generation_defaults.cfg_scale,
                seed=seed,
                slash_command="video",
                sampler=generation_defaults.sampler,
                scheduler=generation_defaults.scheduler,
                fps=generation_defaults.fps,
                filename=await process_attachment(input_file, interaction) if input_file != None else None,
                style_prompt=generation_defaults.style_prompt,
                negative_style_prompt=generation_defaults.negative_style_prompt,
                video_width=generation_defaults.video_width,
                video_length=generation_defaults.video_length,
            )
            await self._do_request(
                interaction,
                f'🎥{interaction.user.mention} asked me to imagine "{prompt}" with WAN! {random.choice(generation_messages)} 🎥',
                f'{interaction.user.mention} asked me to imagine "{prompt}" with WAN! {random.choice(completion_messages)} 🎥',
                "video",
                params,
            )

    async def _do_request(
            self,
            interaction: discord.Interaction,
            intro_message: str,
            completion_message: str,
            command_name: str,
            params: ImageWorkflow,
    ):
        await interaction.response.defer()

        try:
            if should_filter(params.prompt):
                logger.info(
                    "Prompt or negative prompt contains a blocked word, not generating image. Prompt: %s, Negative Prompt: %s",
                    params.prompt,
                    params.negative_prompt,
                )
                await interaction.followup.send(
                    f"The prompt {params.prompt} or negative prompt {params.negative_prompt} contains a blocked word, not generating image.",
                    ephemeral=True,
                )
                return

            await interaction.followup.send(intro_message)

            if params.seed is None:
                params.seed = random.randint(0, 999999999999999)

            if params.filename2 is not None:
                params.workflow_type = WorkflowType.image_mashup

            from src.comfy_workflows import do_workflow

            images = await do_workflow(params, interaction)

            if images is None or len(images) == 0:
                return

            final_message = f"{completion_message}\n Seed: {params.seed}"

            if params.use_llm:
                final_message += f"\nenhanced prompt: `{params.prompt}`"

            params.use_llm = False

            file_name = get_filename(interaction, params)

            fname = f"{file_name}.gif" if "GIF" in images[0].format else f"{file_name}.png"

            collage_path = create_collage(images, params)

            is_nsfw = False
            if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True":
                is_nsfw = check_nsfw(collage_path, params.prompt)

            buttons = Buttons(params, images, interaction.user, is_nsfw, command=command_name)

            await interaction.channel.send(content=final_message, file=discord.File(fp=collage_path, filename=fname, spoiler=is_nsfw), view=buttons)
        except Exception as e:
            logger.exception("Error generating image: %s for command %s with params %s", e, command_name, params)
            await interaction.channel.send(f"{interaction.user.mention} `Error generating image: {e} for command {command_name}`")


class SDXLCommand(ImageGenCommands):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree)
        self.command_name = command_name
        self.command_descs = SDXL_ARG_DESCS
        self.command_choices = SDXL_ARG_CHOICES
        self.model_type = ModelType.SDXL

    def add_commands(self):
        @self.tree.command(name=self.command_name, description=f"Generate an image using {self.command_name.upper()}")
        @app_commands.describe(**self.command_descs)
        @app_commands.choices(**self.command_choices)
        async def slash_command(
                interaction: discord.Interaction,
                prompt: str,
                negative_prompt: str = None,
                model: str = None,
                lora: Choice[str] = None,
                lora_strength: float = 1.0,
                lora2: Choice[str] = None,
                lora_strength2: float = 1.0,
                aspect_ratio: str = None,
                num_steps: Range[int, 1, MAX_STEPS] = None,
                cfg_scale: Range[float, 1.0, MAX_CFG] = None,
                seed: int = None,
                input_file: Attachment = None,
                mashup_image: Attachment = None,
                denoise_strength: Range[float, 0.01, 1.0] = None,
                inpainting_prompt: str = None,
                inpainting_detection_threshold: Range[int, 0, 255] = None,
                style_prompt: Optional[str] = None,
                negative_style_prompt: Optional[str] = None,
                use_llm: Optional[bool] = None,
                mashup_image_strength: Optional[float] = None,
                mashup_inputimage_strength: Optional[float] = None,
                controlnet_type: Optional[str] = None,
                controlnet_strength: Optional[float] = None,
                controlnet_start_percent: Optional[float] = None,
                controlnet_end_percent: Optional[float] = None,
        ):
            if input_file is not None:
                fp = await process_attachment(input_file, interaction)
                if fp is None:
                    return

            if mashup_image is not None:
                fp2 = await process_attachment(mashup_image, interaction)
                if fp2 is None:
                    return

            defaults = COMMAND_DEFAULTS[self.command_name]

            params = ImageWorkflow(
                model_type=self.model_type,
                workflow_type=WorkflowType.txt2img if input_file is None or controlnet_type is not None else WorkflowType.img2img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=model or defaults.model,
                loras=unpack_choices(lora, lora2),
                lora_strengths=[lora_strength, lora_strength2],
                dimensions=sd_aspect_ratios[aspect_ratio] if aspect_ratio else sd_aspect_ratios[defaults.dimensions],
                sampler=defaults.sampler,
                num_steps=num_steps or defaults.num_steps,
                cfg_scale=cfg_scale or defaults.cfg_scale,
                denoise_strength=defaults.denoise_strength or denoise_strength if input_file is not None and controlnet_type is None else 1.0,
                batch_size=defaults.batch_size,
                seed=seed,
                filename=fp if input_file is not None else None,
                slash_command=self.command_name,
                inpainting_prompt=inpainting_prompt,
                inpainting_detection_threshold=inpainting_detection_threshold or defaults.inpainting_detection_threshold,
                clip_skip=defaults.clip_skip,
                filename2=fp2 if mashup_image is not None else None,
                use_accelerator_lora=defaults.use_accelerator_lora,
                accelerator_lora_name=(
                    defaults.accelerator_lora_name
                    if defaults.use_accelerator_lora
                    else None
                ),
                scheduler=defaults.scheduler,
                style_prompt=style_prompt or defaults.style_prompt,
                negative_style_prompt=negative_style_prompt or defaults.negative_style_prompt,
                vae=None,
                detailing_controlnet=defaults.detailing_controlnet,
                use_llm=use_llm or (bool(config["LLM"]["use_llm"]) and self.command_name == "imagine"),
                use_align_your_steps=bool(defaults.use_align_your_steps),
                use_teacache=bool(defaults.use_teacache),
                use_triton=bool(defaults.use_triton),
                use_tensorrt=defaults.use_tensorrt or False,
                tensorrt_model=defaults.tensorrt_model,
                mashup_image_strength=mashup_image_strength,
                mashup_inputimage_strength=mashup_inputimage_strength,
                controlnet_model=defaults.controlnet_model,
                controlnet_type=controlnet_type,
                controlnet_strength=controlnet_strength or defaults.controlnet_strength,
                controlnet_start_percent=controlnet_start_percent or defaults.controlnet_start_percent,
                controlnet_end_percent=controlnet_end_percent or defaults.controlnet_end_percent,
            )

            await self._do_request(
                interaction,
                f'🖌️{interaction.user.mention} asked me to imagine "{prompt}" using {self.command_name.upper()}! {random.choice(generation_messages)} 🖌️',
                f'🖌️ {interaction.user.mention} asked me to imagine "{prompt}" using {self.command_name.upper()}! {random.choice(completion_messages)}. 🖌️',
                self.command_name,
                params,
            )


class PonyXLCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "pony")
        self.command_descs = PONY_ARG_DESCS
        self.command_choices = PONY_ARG_CHOICES
        self.model_type = ModelType.PONY


class SD3Command(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "sd3")
        self.command_descs = SD3_ARG_DESCS
        self.command_choices = SD3_ARG_CHOICES
        self.model_type = ModelType.SD3


class FluxCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "flux")
        self.command_descs = FLUX_ARG_DESCS
        self.command_choices = FLUX_ARG_CHOICES
        self.model_type = ModelType.FLUX
        
class EditCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "edit")
        self.command_descs = FLUX_ARG_DESCS
        self.command_choices = FLUX_ARG_CHOICES
        self.model_type = ModelType.FLUX_KONTEXT


class ImagineCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "imagine")
        self.command_descs = FLUX_ARG_DESCS
        self.command_choices = FLUX_ARG_CHOICES
        self.model_type = ModelType.FLUX


class CascadeCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "cascade")
        self.command_descs = CASCADE_ARG_DESCS
        self.command_choices = CASCADE_ARG_CHOICES
        self.model_type = ModelType.CASCADE


class LegacyCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "legacy")
        self.command_descs = LEGACY_ARG_DESCS
        self.command_choices = LEGACY_ARG_CHOICES
        self.model_type = ModelType.SD15
