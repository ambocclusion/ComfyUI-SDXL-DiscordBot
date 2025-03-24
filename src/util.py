import configparser
import os

from discord import Interaction, Attachment
from src.image_gen import ImageWorkflow


def read_config():
    config = configparser.ConfigParser()
    config.read("config.properties", encoding='utf8')
    return config


def generate_default_config():
    config = configparser.ConfigParser()
    config["BOT"] = {"TOKEN": "YOUR_DEFAULT_DISCORD_BOT_TOKEN"}
    config["LOCAL"] = {"SERVER_ADDRESS": "YOUR_COMFYUI_URL"}
    with open("../config.properties", "w") as configfile:
        config.write(configfile)


def setup_config():
    if not os.path.exists("../config.properties"):
        generate_default_config()

    if not os.path.exists("../out"):
        os.makedirs("../out")

    config = read_config()
    token = (
        os.environ["DISCORD_APP_TOKEN"]
        if "DISCORD_APP_TOKEN" in os.environ
        else config["BOT"]["TOKEN"]
    )
    return token


def should_filter(positive_prompt: str) -> bool:
    positive_prompt = positive_prompt or ""

    config = read_config()
    word_list = config["BLOCKED_WORDS"]["WORDS"].split(",")
    if word_list is None:
        return False
    for word in word_list:
        if word.lower() in positive_prompt.lower():
            return True
    return False


def unpack_choices(*args):
    return [x is not None and x.value or None for x in args]


def get_filename(interaction: Interaction, params: ImageWorkflow):
    return f"{interaction.user.name}_{params.prompt[:10]}_{params.seed}"


def build_command(params: ImageWorkflow):
    try:
        command = f"/{params.slash_command}"
        command += f" prompt:{params.prompt}"
        if params.negative_prompt:
            command += f" negative_prompt:{params.negative_prompt}"
        command += f" seed:{params.seed}"
        command += f" model:{params.model.replace('.safetensors', '')}"
        command += f" sampler:{params.sampler}"
        command += f" num_steps:{params.num_steps}"
        command += f" cfg_scale:{params.cfg_scale}"
        if len(params.loras) != 0:
            for i, lora in enumerate(params.loras):
                if lora is None or lora == "None":
                    continue
                command += f" lora{i > 0 and i + 1 or ''}:{str(lora).replace('.safetensors', '')}"
                command += f" lora_strength{i > 0 and i + 1 or ''}:{params.lora_strengths[i]}"
        if params.filename is not None:
            command += f" input_file:[Attachment]"
        if params.denoise_strength is not None:
            command += f" denoise_strength:{params.denoise_strength}"
        return command
    except Exception as e:
        print(e)
        return ""


async def process_attachment(attachment: Attachment, interaction: Interaction):
    if attachment.content_type != "image/png" and attachment.content_type != "image/jpeg":
        await interaction.response.send_message("Error: Please upload a PNG or JPEG image", ephemeral=True)
        return None

    os.makedirs("../input", exist_ok=True)

    fp = f"./input/{attachment.filename}"
    await attachment.save(fp)

    if attachment.width > 1024 or attachment.height > 1024:
        from PIL import Image
        img = Image.open(fp)
        if img.width > img.height:
            img = img.resize((1024, int(img.height * 1024 / img.width)))
        else:
            img = img.resize((int(img.width * 1024 / img.height), 1024))
        img.save(fp)

    if attachment.width < 1024 and attachment.height < 1024:
        from PIL import Image
        img = Image.open(fp)
        scaling_factor = max(1024 / attachment.width, 1024 / attachment.height)

        img = img.resize((int(attachment.width * scaling_factor), int(attachment.height * scaling_factor)))
        img.save(fp)

    return os.path.abspath(fp)

def get_server_address():
    import configparser
    config = configparser.ConfigParser()
    config.read("config.properties", encoding="utf8")

    if config["BOT"]["USE_EMBEDDED_COMFY"].lower() == "true":
        return f"http://localhost:{config['EMBEDDED']['SERVER_PORT']}"
    else:
        return config['LOCAL']['SERVER_ADDRESS']

def get_loras_from_prompt(prompt: str):
    import re
    loras = re.findall(r"<lora:(.*?):(.*?)>", prompt)
    return [[lora[0], float(lora[1])] for lora in loras]

def get_workflow(image, image_workflow: ImageWorkflow = None):
    info = image.info
    pnginfo = {}
    if info is not None:
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        for key, value in info.items():
            pnginfo.add_text(key, value)
    if image_workflow is not None:
        pnginfo.add_text("parameters", str(image_workflow))
    return pnginfo