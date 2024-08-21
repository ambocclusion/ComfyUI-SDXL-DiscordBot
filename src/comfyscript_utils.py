import asyncio

from comfy_script.runtime import load
from src.util import get_server_address


async def server_is_started() -> bool:
    # TODO: this sucks, redo it
    while True:
        try:
            load(get_server_address())
            break
        except:
            await asyncio.sleep(3)
    # do api call to check if server is started
    from comfy_script.runtime import client
    try:
        print(len(client.get_nodes_info()))
        return len(client.get_nodes_info()) > 0
    except:
        return False

def get_models():
    from comfy_script.runtime.nodes import Checkpoints
    models = [model.value for model in Checkpoints]

    try:
        from comfy_script.runtime.nodes import UNETs
        from comfy_script.runtime.nodes import UnetLoaderGGUF
        models.extend([model.value for model in UNETs])
        models.extend([model.value for model in UnetLoaderGGUF.unet_name])
    except:
        pass

    return models

def get_loras():
    from comfy_script.runtime.nodes import Loras
    loras = [lora.value for lora in Loras]
    return loras

def get_samplers():
    from comfy_script.runtime.nodes import Samplers
    samplers = [sampler.value for sampler in Samplers]
    return samplers

def get_schedulers():
    from comfy_script.runtime.nodes import Schedulers
    schedulers = [scheduler.value for scheduler in Schedulers]
    return schedulers

def get_tortoise_voices():
    from comfy_script.runtime.nodes import TortoiseTTSGenerate
    voices = [voice.value for voice in TortoiseTTSGenerate.voice]
    return voices