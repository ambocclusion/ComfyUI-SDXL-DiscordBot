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
        @self.tree.command(name="legacy", description="Generate an image based on input text")
        @app_commands.describe(**IMAGINE_ARG_DESCS)
        @app_commands.choices(**IMAGINE_ARG_CHOICES)
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
            sampler: str = None,
            scheduler: str = None,
            num_steps: Range[int, 1, MAX_STEPS] = None,
            cfg_scale: Range[float, 1.0, MAX_CFG] = None,
            seed: int = None,
            input_file: Attachment = None,
            denoise_strength: Range[float, 0.01, 1.0] = None,
            inpainting_prompt: str = None,
            inpainting_detection_threshold: Range[int, 0, 255] = None,
            clip_skip: Range[int, -2, -1] = None,
            use_llm: Optional[bool] = None,
        ):
            if input_file is not None:
                fp = await process_attachment(input_file, interaction)
                if fp is None:
                    return

            dimensions = sd_aspect_ratios[aspect_ratio] if aspect_ratio else sd_aspect_ratios[SD15_GENERATION_DEFAULTS.dimensions]
            dimensions = (dimensions[0] / 2, dimensions[1] / 2)

            params = ImageWorkflow(
                ModelType.SD15,
                WorkflowType.txt2img if input_file is None else WorkflowType.img2img,
                prompt,
                negative_prompt,
                model or SD15_GENERATION_DEFAULTS.model,
                unpack_choices(lora, lora2),
                [lora_strength, lora_strength2],
                dimensions,
                sampler or SD15_GENERATION_DEFAULTS.sampler,
                num_steps or SD15_GENERATION_DEFAULTS.num_steps,
                cfg_scale or SD15_GENERATION_DEFAULTS.cfg_scale,
                seed=seed,
                slash_command="legacy",
                filename=fp if input_file is not None else None,
                denoise_strength=denoise_strength or SD15_GENERATION_DEFAULTS.denoise_strength if input_file is not None else 1.0,
                batch_size=SD15_GENERATION_DEFAULTS.batch_size,
                inpainting_prompt=inpainting_prompt,
                inpainting_detection_threshold=inpainting_detection_threshold or SD15_GENERATION_DEFAULTS.inpainting_detection_threshold,
                clip_skip=clip_skip or SD15_GENERATION_DEFAULTS.clip_skip,
                use_llm=use_llm or False,
                scheduler=scheduler or SD15_GENERATION_DEFAULTS.scheduler,
                use_tensorrt=SD15_GENERATION_DEFAULTS.use_tensorrt,
                tensorrt_model=SD15_GENERATION_DEFAULTS.tensorrt_model,
                use_accelerator_lora=SD15_GENERATION_DEFAULTS.use_accelerator_lora,
            )
            await self._do_request(
                interaction,
                f'🖼️ {interaction.user.mention} asked me to imagine "{prompt}"! {random.choice(generation_messages)} 🖼️',
                f'{interaction.user.mention} asked me to imagine "{prompt}"! {random.choice(completion_messages)}',
                "imagine",
                params,
            )

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
        async def slash_command(
            interaction: discord.Interaction,
            prompt: str,
            negative_prompt: str = None,
            cfg_scale: Range[float, 1.0, MAX_CFG] = None,
            input_file: Attachment = None,
            seed: int = None,
        ):
            if input_file is not None and input_file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                await interaction.response.send_message(
                    f"{interaction.user.mention} `Only PNG, JPG, and JPEG images are supported for video generation`",
                    ephemeral=True,
                )
                return

            params = ImageWorkflow(
                ModelType.VIDEO,
                WorkflowType.wan if input_file == None else WorkflowType.image_wan,
                prompt,
                negative_prompt,
                model=WAN_GENERATION_DEFAULTS.model if input_file == None else IMAGE_WAN_GENERATION_DEFAULTS.model,
                num_steps=WAN_GENERATION_DEFAULTS.num_steps if input_file == None else IMAGE_WAN_GENERATION_DEFAULTS.num_steps,
                cfg_scale=cfg_scale or WAN_GENERATION_DEFAULTS.cfg_scale,
                seed=seed,
                slash_command="video",
                sampler=WAN_GENERATION_DEFAULTS.sampler if input_file == None else IMAGE_WAN_GENERATION_DEFAULTS.sampler,
                scheduler=WAN_GENERATION_DEFAULTS.scheduler if input_file == None else IMAGE_WAN_GENERATION_DEFAULTS.scheduler,
                fps=WAN_GENERATION_DEFAULTS.fps if input_file == None else IMAGE_WAN_GENERATION_DEFAULTS.fps,
                filename=await process_attachment(input_file, interaction) if input_file != None else None,
                style_prompt = WAN_GENERATION_DEFAULTS.style_prompt,
                negative_style_prompt = WAN_GENERATION_DEFAULTS.negative_style_prompt
            )
            await self._do_request(
                interaction,
                f'🎥{interaction.user.mention} asked me to imagine "{prompt}" with WAN! {random.choice(generation_messages)} 🎥',
                f'{interaction.user.mention} asked me to imagine "{prompt}" with WAN! {random.choice(completion_messages)} 🎥',
                "video",
                params,
            )

        @self.tree.command(name="cascade", description="Use Stable Cascade to generate an image")
        @app_commands.describe(**CASCADE_ARG_DESCS)
        @app_commands.choices(**CASCADE_ARG_CHOICES)
        async def slash_command(
            interaction: discord.Interaction,
            prompt: str,
            negative_prompt: str = None,
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
            clip_skip: Range[int, -2, -1] = None,
            use_llm: Optional[bool] = None,
        ):
            if input_file is not None:
                fp = await process_attachment(input_file, interaction)
                if fp is None:
                    return

            if mashup_image is not None:
                fp2 = await process_attachment(mashup_image, interaction)
                if fp2 is None:
                    return

            params = ImageWorkflow(
                ModelType.CASCADE,
                WorkflowType.txt2img if input_file is None else WorkflowType.img2img,
                prompt,
                negative_prompt,
                CASCADE_GENERATION_DEFAULTS.model,
                loras=unpack_choices(lora, lora2),
                lora_strengths=[lora_strength, lora_strength2],
                dimensions=sd_aspect_ratios[aspect_ratio] if aspect_ratio else sd_aspect_ratios[CASCADE_GENERATION_DEFAULTS.dimensions],
                sampler=CASCADE_GENERATION_DEFAULTS.sampler,
                num_steps=num_steps or CASCADE_GENERATION_DEFAULTS.num_steps,
                cfg_scale=cfg_scale or CASCADE_GENERATION_DEFAULTS.cfg_scale,
                denoise_strength=denoise_strength or CASCADE_GENERATION_DEFAULTS.denoise_strength,
                batch_size=CASCADE_GENERATION_DEFAULTS.batch_size,
                seed=seed,
                filename=fp if input_file is not None else None,
                filename2=fp2 if mashup_image is not None else None,
                inpainting_prompt=inpainting_prompt,
                inpainting_detection_threshold=inpainting_detection_threshold or CASCADE_GENERATION_DEFAULTS.inpainting_detection_threshold,
                clip_skip=clip_skip or CASCADE_GENERATION_DEFAULTS.clip_skip,
                use_llm=use_llm or False,
            )

            await self._do_request(
                interaction,
                f'🤖️ {interaction.user.mention} asked me to imagine "{prompt}" using Stable Cascade! {random.choice(generation_messages)} 🤖️',
                f'🤖️ {interaction.user.mention} asked me to imagine "{prompt}" using Stable Cascade! {random.choice(completion_messages)} 🤖️',
                "cascade",
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
            sampler: str = None,
            scheduler: str = None,
            num_steps: Range[int, 1, MAX_STEPS] = None,
            cfg_scale: Range[float, 1.0, MAX_CFG] = None,
            seed: int = None,
            input_file: Attachment = None,
            mashup_image: Attachment = None,
            denoise_strength: Range[float, 0.01, 1.0] = None,
            inpainting_prompt: str = None,
            inpainting_detection_threshold: Range[int, 0, 255] = None,
            clip_skip: Range[int, -2, -1] = None,
            use_accelerator_lora: Optional[bool] = None,
            style_prompt: Optional[str] = None,
            negative_style_prompt: Optional[str] = None,
            use_llm: Optional[bool] = None,
            mashup_image_strength: Optional[float] = None,
            mashup_inputimage_strength: Optional[float] = None
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
                self.model_type,
                WorkflowType.txt2img if input_file is None else WorkflowType.img2img,
                prompt,
                negative_prompt,
                model or defaults.model,
                unpack_choices(lora, lora2),
                [lora_strength, lora_strength2],
                dimensions=sd_aspect_ratios[aspect_ratio] if aspect_ratio else sd_aspect_ratios[defaults.dimensions],
                batch_size=defaults.batch_size,
                sampler=sampler or defaults.sampler,
                num_steps=num_steps or defaults.num_steps,
                cfg_scale=cfg_scale or defaults.cfg_scale,
                seed=seed,
                slash_command=self.command_name,
                filename=fp if input_file is not None else None,
                filename2=fp2 if mashup_image is not None else None,
                denoise_strength=denoise_strength or defaults.denoise_strength if input_file is not None else 1.0,
                inpainting_prompt=inpainting_prompt,
                inpainting_detection_threshold=inpainting_detection_threshold or defaults.inpainting_detection_threshold,
                clip_skip=clip_skip or defaults.clip_skip,
                use_accelerator_lora=use_accelerator_lora or defaults.use_accelerator_lora,
                accelerator_lora_name=(
                    defaults.accelerator_lora_name
                    if use_accelerator_lora or (use_accelerator_lora is None and defaults.use_accelerator_lora)
                    else None
                ),
                scheduler=scheduler or defaults.scheduler,
                style_prompt=style_prompt or defaults.style_prompt,
                negative_style_prompt=negative_style_prompt or defaults.negative_style_prompt,
                detailing_controlnet=defaults.detailing_controlnet,
                use_llm=use_llm or (bool(config["LLM"]["use_llm"]) and self.command_name == "imagine"),
                use_align_your_steps=bool(defaults.use_align_your_steps),
                use_tensorrt=defaults.use_tensorrt or False,
                tensorrt_model=defaults.tensorrt_model,
                mashup_image_strength=mashup_image_strength,
                mashup_inputimage_strength=mashup_inputimage_strength
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

class ImagineCommand(SDXLCommand):
    def __init__(self, tree: discord.app_commands.CommandTree, command_name: str):
        super().__init__(tree, "imagine")
        self.command_descs = FLUX_ARG_DESCS
        self.command_choices = FLUX_ARG_CHOICES
        self.model_type = ModelType.FLUX
