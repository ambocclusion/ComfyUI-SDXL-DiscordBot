import configparser
import contextlib
import os
import random
import wave
from typing import Protocol, runtime_checkable

from src.audio_gen.audio_workflow import AudioWorkflowParams, AudioModelType

from comfy_script.runtime import *
from src.util import get_server_address
load(get_server_address())

from comfy_script.runtime.nodes import *

config = configparser.ConfigParser()
config.read("config.properties", encoding="utf8")


def get_data(results):
    output_directory = config["LOCAL"]["COMFY_ROOT_DIR"]
    audio_filenames = []
    video_filenames = []
    video_data = []

    for audio in results._output.get("audio", []):
        filename = os.path.join(output_directory, audio["type"], audio["subfolder"], audio["filename"])
        audio_filenames.append(filename)

    for video in results._output.get("video", []):
        filename = os.path.join(output_directory, video["type"], video["subfolder"], video["filename"])
        video_filenames.append(filename)
        with open(filename, "rb") as file:
            video_data.append(file.read())

    return (video_data, video_filenames, audio_filenames)


def comfy_workflow(func):
    """decorator which wraps a function in a comfyscript workflow and extracts its outputs."""
    async def wrapper(self, *args, **kwargs):
        with Workflow():
            results = func(self, *args, **kwargs)
        return get_data(await results._wait())
    return wrapper


def get_sound_duration(snd_filename):
    with contextlib.closing(wave.open(snd_filename, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def make_spectrogram_webm(audio):
    spectrogram_image = SpectrogramImage(audio, 1024, 256, 1024, 0.4, normalized=True)
    spectrogram_image = ImageScale(spectrogram_image, ImageScale.upscale_method.lanczos, 512, 128, "disabled")
    return CombineImageWithAudio(spectrogram_image, audio, CombineImageWithAudio.file_format.webm, "ComfyUI")


class AudioWorkflow:
    def __init__(self, params: AudioWorkflowParams):
        self.params = params

    """the following Protocols define a weak 'contract' for the AudioWorkflow that enables
    the use of type introspection to check the capabilities of a workflow according to whether
    or not its class implements particular named methods. these are defined within the scope of
    this class only for organizational purposes.
    """

    @runtime_checkable
    class Generateable(Protocol):
        def generate(self): ...

    @runtime_checkable
    class Extendable(Protocol):
        def extend(self): ...

    @runtime_checkable
    class Remixable(Protocol):
        def remix(self): ...


class MusicgenWorkflow(AudioWorkflow):
    @comfy_workflow
    def generate(self):
        params = self.params
        model, sr = MusicgenLoader()
        raw_audio = MusicgenGenerate(model, params.prompt, 4, params.duration, params.cfg_scale, params.top_k, params.top_p, params.temperature, params.seed or random.randint(0, 2**32 - 1))
        return make_spectrogram_webm(raw_audio)

    @comfy_workflow
    def extend(self):
        params = self.params
        full_dur = get_sound_duration(params.snd_filename)

        model, model_sr = MusicgenLoader()
        full_audio = LoadAudio(params.snd_filename)
        full_audio = ConvertAudio(full_audio, model_sr, 1)

        # get last 10 seconds (at most) of audio to extend
        # TODO: configurable conditioning duration
        start_s = max(full_dur - 10.0, 0.0)
        end_s = full_dur
        in_dur = end_s - start_s
        in_audio = ClipAudioRegion(full_audio, start_s, end_s)

        out_dur = in_dur + (params.duration or 10)
        out_audio = MusicgenGenerate(model, params.prompt, 4, out_dur, params.cfg_scale, params.top_k, params.top_p, params.temperature, params.seed or random.randint(0, 2**32 - 1), in_audio)

        # cut prefix section from output to isolate extension, then combine with original audio
        out_audio = ClipAudioRegion(out_audio, in_dur, out_dur)
        combined_audio = ConcatAudio(full_audio, out_audio)
        return make_spectrogram_webm(combined_audio)


class TortoiseTTSWorkflow(AudioWorkflow):
    @comfy_workflow
    def generate(self):
        params = self.params
        model, sr = TortoiseTTSLoader(True, False, False, False)
        out_audio = TortoiseTTSGenerate(model, params.voice, params.prompt, 4, 8, 8, 0.3, 2, 4, 0.8, 300, 0.70, 10, True, 2, 1, params.seed or random.randint(0, 2**32 - 1))
        return make_spectrogram_webm(out_audio)


class TortoiseMusicgenWorkflow(MusicgenWorkflow):
    @comfy_workflow
    def generate(self):
        params = self.params
        tts_model, tts_sr = TortoiseTTSLoader(True, False, False, False)
        tts_audio = TortoiseTTSGenerate(tts_model, params.voice, params.prompt, 4, 8, 8, 0.3, 2, 4, 0.8, 300, 0.70, 10, True, 2, 1, params.seed or random.randint(0, 2**32 - 1))
        model, sr = MusicgenLoader()
        tts_audio = ConvertAudio(tts_audio, tts_sr, sr, 1)
        out_audio = MusicgenGenerate(model, params.secondary_prompt, 4, 15, params.cfg_scale, params.top_k, params.top_p, params.temperature, params.seed or random.randint(0, 2**32 - 1), tts_audio)
        return make_spectrogram_webm(out_audio)


class AceStepWorkflow(AudioWorkflow):
    def _load_model(self):
        model, clip, vae = CheckpointLoaderSimple(ckpt_name=CheckpointLoaderSimple.ckpt_name.ace_step_v1_3_5b)
        model = ModelSamplingSD3(model=model, shift=5.0)
        op = LatentOperationTonemapReinhard(1.0)  # TODO: controls vocal volume in comfyui example workflow; make it configurable?
        model = LatentApplyOperationCFG(model=model, operation=op)
        return model, clip, vae

    def _generate_with_latent(self, model, clip, vae, latent):
        params = self.params
        positive = TextEncodeAceStepAudio(clip=clip, tags=params.prompt, lyrics=params.secondary_prompt or "", lyrics_strength=params.lyrics_strength)
        if params.negative_prompt is not None:
            negative = TextEncodeAceStepAudio(clip=clip, tags=params.negative_prompt, lyrics="", lyrics_strength=0.0)
        else:
            negative = ConditioningZeroOut(conditioning=positive)
        latent = KSampler(
            model=model, seed=params.seed or random.randint(0, 2**32 - 1), steps=params.num_steps, cfg=params.cfg_scale, positive=positive, negative=negative, latent_image=latent, denoise=params.denoise_strength
        )
        out_audio = VAEDecodeAudio(samples=latent, vae=vae)
        return make_spectrogram_webm(out_audio)

    @comfy_workflow
    def generate(self):
        params = self.params
        model, clip, vae = self._load_model()
        latent = EmptyAceStepLatentAudio(seconds=params.duration, batch_size=params.batch_size or 1)
        return self._generate_with_latent(model, clip, vae, latent)

    @comfy_workflow
    def remix(self):
        params = self.params
        model, clip, vae = self._load_model()
        audio = LoadAudio(params.snd_filename)
        latent = VAEEncodeAudio(audio, vae)
        return self._generate_with_latent(model, clip, vae, latent)


model_type_to_workflow = {
    AudioModelType.MUSICGEN: MusicgenWorkflow,
    AudioModelType.ACESTEP: AceStepWorkflow,
    AudioModelType.TORTOISE: TortoiseTTSWorkflow,
    AudioModelType.TORTOISE_MUSICGEN: TortoiseMusicgenWorkflow,
}