from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.defaults import get_default_from_config as get_default_str


class AudioModelType(Enum):
    MUSICGEN = "musicgen"
    ACESTEP = "acestep"
    TORTOISE = "tortoise"
    TORTOISE_MUSICGEN = "sing"


class AudioWorkflowType(Enum):
    GENERATE = "generate"
    EXTEND = "extend"
    REMIX = "remix"


@dataclass
class AudioWorkflowParams:
    model_type: AudioModelType
    workflow_type: AudioWorkflowType

    prompt: str
    negative_prompt: Optional[str] = None
    secondary_prompt: Optional[str] = None

    voice: Optional[str] = None

    duration: Optional[float] = None

    num_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    temperature: Optional[float] = None

    batch_size: Optional[int] = None
    seed: Optional[int] = None

    snd_filename: Optional[list[str]] = None

    denoise_strength: Optional[float] = None
    lyrics_strength: Optional[float] = None


def get_default(section: str, name: str, cast: Callable, default: Optional[Any] = None):
    value = get_default_str(section, name, default)
    return cast(value) if value is not None else None


MUSICGEN_DEFAULTS = AudioWorkflowParams(
    AudioModelType.MUSICGEN,
    AudioWorkflowType.GENERATE,
    None,
    duration=get_default("MUSICGEN_DEFAULTS", "DURATION", float, 30.0),
    cfg_scale=get_default("MUSICGEN_DEFAULTS", "CFG", float, 4.0),
    top_k=get_default("MUSICGEN_DEFAULTS", "TOP_K", int, 250),
    top_p=get_default("MUSICGEN_DEFAULTS", "TOP_P", float, 0.0),
    temperature=get_default("MUSICGEN_DEFAULTS", "TEMPERATURE", float, 1),
)


ACESTEP_DEFAULTS = AudioWorkflowParams(
    AudioModelType.ACESTEP,
    AudioWorkflowType.GENERATE,
    None,
    duration=get_default("ACESTEP_DEFAULTS", "DURATION", float, 180.0),
    cfg_scale=get_default("ACESTEP_DEFAULTS", "CFG", float, 5.0),
    num_steps=get_default("ACESTEP_DEFAULTS", "NUM_STEPS", int, 60),
    denoise_strength=1.0,
    lyrics_strength=get_default("ACESTEP_DEFAULTS", "LYRICS_STRENGTH", float, 0.99),
)


TORTOISE_DEFAULTS = AudioWorkflowParams(
    AudioModelType.TORTOISE,
    AudioWorkflowType.GENERATE,
    None,
    voice=get_default("TORTOISE_DEFAULTS", "VOICE", str, "random"),
    top_p=get_default("TORTOISE_DEFAULTS", "TOP_P", float, 0.8),
    temperature=get_default("TORTOISE_DEFAULTS", "TEMPERATURE", float, 0.3),
)
