import logging
import json
import random
from copy import deepcopy
from io import BytesIO
from typing import Optional

import discord
from discord import app_commands
from discord.app_commands import Range, Choice

from src.audio_gen.audio_gen import (
    model_type_to_workflow
)
from src.audio_gen.audio_workflow import (
    AudioModelType,
    ACESTEP_DEFAULTS,
    MUSICGEN_DEFAULTS,
    TORTOISE_DEFAULTS,
)
from src.audio_gen.ui.audio_buttons import AudioButtons

logger = logging.getLogger("bot")

voice_choice_map = json.load(open("data/voice_selections.json"))
TORTOISE_VOICE_CHOICES = [Choice(name=v, value=k) for k, v in voice_choice_map.items()]


class SoundCommand:
    async def _do_request(
        self,
        interaction: discord.Interaction,
        intro_message,
        completion_message,
        params,
    ):
        await interaction.response.send_message(intro_message)

        params.seed = params.seed or random.randint(0, 999999999999999)

        workflow = model_type_to_workflow[params.model_type](params)
        videos, _, sound_fnames = await workflow.generate()

        final_message = f"{completion_message}\n Seed: {params.seed}"
        buttons = AudioButtons(params, sound_fnames)
        files = [discord.File(BytesIO(vid), filename=f"sound_{i}.webm") for i, vid in enumerate(videos)]
        await interaction.channel.send(content=final_message, files=files, view=buttons)


class MusicGenCommand(SoundCommand):
    def __init__(self, tree: discord.app_commands.CommandTree):
        self.tree = tree

    def add_commands(self):
        @self.tree.command(name="musicgen", description="Generate music from text using Musicgen.")
        async def music_command(
            interaction: discord.Interaction,
            prompt: str,
            duration: Range[float, 5.0, 10.0] = None,
            cfg_scale: Range[float, 0.0, 100.0] = None,
            top_k: Range[int, 0, 1000] = None,
            top_p: Range[float, 0.0, 1.0] = None,
            temperature: Range[float, 1e-3, 10.0] = None,
            seed: int = None,
        ):
            params = deepcopy(MUSICGEN_DEFAULTS)
            params.prompt = prompt
            params.duration = duration or params.duration
            params.cfg_scale = cfg_scale or params.cfg_scale
            params.top_k = top_k or params.top_k
            params.top_p = top_p or params.top_p
            params.temperature = temperature or params.temperature
            params.seed = seed

            msg_prefix = f'{interaction.user.mention} asked me to make music that sounds like "{prompt}", '
            await self._do_request(
                interaction,
                msg_prefix + "this shouldn't take too long...",
                msg_prefix + "here is what I made for them.",
                params,
            )

        @self.tree.command(name="acestep", description="Generate music from text using ACE-Step.")
        async def acestep_command(
            interaction: discord.Interaction,
            prompt: str,
            lyrics: Optional[str],
            negative_prompt: Optional[str],
            duration: Range[float, 10.0, 213.0] = None,
            cfg_scale: Range[float, 1.0, 100.0] = None,
            num_steps: Range[int, 1, 60] = None,
            lyrics_strength: Range[float, 0.0, 10.0] = None,
            seed: int = None,
        ):
            params = deepcopy(ACESTEP_DEFAULTS)
            params.prompt = prompt
            params.negative_prompt = negative_prompt
            if lyrics is not None:
                params.secondary_prompt = '\n'.join([x.strip() for x in lyrics.split("/")])
            params.duration = duration or params.duration
            params.cfg_scale = cfg_scale or params.cfg_scale
            params.num_steps = num_steps or params.num_steps
            params.lyrics_strength = lyrics_strength or params.lyrics_strength
            params.seed = seed

            msg_prefix = f'{interaction.user.mention} asked me to make music that sounds like "{prompt}", '
            await self._do_request(
                interaction,
                msg_prefix + "this shouldn't take too long...",
                msg_prefix + "here is what I made for them.",
                params,
            )

class SpeechGenCommand(SoundCommand):
    def __init__(self, tree: discord.app_commands.CommandTree):
        self.tree = tree

    def add_commands(self):
        @self.tree.command(name="tortoise", description="Generate speech from text using TorToiSe.")
        @app_commands.choices(voice=TORTOISE_VOICE_CHOICES)
        async def speech_command(
            interaction: discord.Interaction,
            prompt: str,
            voice: str = None,
            top_p: Range[float, 0.0, 1.0] = None,
            temperature: Range[float, 1e-3, 10.0] = None,
            seed: int = None,
        ):
            params = deepcopy(TORTOISE_DEFAULTS)
            params.prompt = prompt
            params.voice = voice or params.voice
            params.top_p = top_p or params.top_p
            params.temperature = temperature or params.temperature
            params.seed = seed

            await self._do_request(
                interaction,
                f'{interaction.user.mention} wants to speak, this shouldn\'t take too long...',
                f'{interaction.user.mention} said "{prompt}".',
                params,
            )

        @self.tree.command(name="sing", description="Sing!")
        @app_commands.choices(voice=TORTOISE_VOICE_CHOICES)
        async def sing_command(
            interaction: discord.Interaction,
            music_prompt: str,
            lyrics: str,
            voice: str = None,
            top_k: Range[int, 0, 1000] = None,
            top_p: Range[float, 0.0, 1.0] = None,
            temperature: Range[float, 1e-3, 10.0] = None,
            seed: int = None,
        ):
            params = deepcopy(MUSICGEN_DEFAULTS)
            params.model_type = AudioModelType.TORTOISE_MUSICGEN
            params.prompt = lyrics
            params.secondary_prompt = music_prompt
            params.voice = voice or params.voice
            params.top_p = top_p or params.top_p
            params.top_k = top_k or params.top_k
            params.temperature = temperature or params.temperature
            params.seed = seed

            await self._do_request(
                interaction,
                f'🎙️{interaction.user.mention} wants to sing, this shouldn\'t take too long...🎙️',
                f'🎙️{interaction.user.mention} sang "{lyrics}".🎙️',
                params,
            )
