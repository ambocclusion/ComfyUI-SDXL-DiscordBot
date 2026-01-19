import configparser
import os
from math import floor

import PIL.Image
import discord
import asyncio

from PIL import Image

from comfy_script.runtime import Workflow, queue
from comfy_script.runtime.nodes import LoadImage
from src.ModelDefinition import ModelDefinition
from src.defaults import MAX_RETRIES
from src.image_gen.ImageWorkflow import ImageWorkflow, WorkflowType, ModelType
from src.image_gen.nsfw_detection import check_nsfw
from src.image_gen.generation_workflows.sd_workflows import UpscaleWorkflow, Lora
from src.util import get_loras_from_prompt

config = configparser.ConfigParser()
config.read("config.properties", encoding="utf8")
comfy_root_directory = config["LOCAL"]["COMFY_ROOT_DIR"]
use_align_your_steps = config["SVD_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"].lower()

loop = None


async def _do_txt2img(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    with Workflow() as wf:
        workflow = model_definition.workflow(params)
        workflow.create_latents()
        workflow.condition_prompts()
        workflow.sample(use_ays=params.use_align_your_steps)
        images, file_names = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction, params.prompt))
    # TODO: Change this .wait function into a `get_results` function instead
    await workflow.wait()
    results = await images
    await results
    if file_names is None:
        image_batch = [await results.get(i) for i in range(params.batch_size)]
        return image_batch
    else:
        await file_names._wait()
        file_name_results = file_names.wait()._output
        image_batch = PIL.Image.open(os.path.join(comfy_root_directory, "output", file_name_results["gifs"][0]["filename"]))
        return [image_batch]


async def _do_img2img(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    with Workflow() as wf:
        workflow = model_definition.workflow(params)
        image_input = LoadImage(params.filename)[0]
        workflow.create_img2img_latents(image_input)
        if params.inpainting_prompt:
            workflow.mask_for_inpainting(image_input)
        workflow.condition_prompts()
        workflow.sample(params.use_align_your_steps)
        images, file_names = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction, params.prompt))
    
    results = await images
    await results
    if file_names is None:
        image_batch = [await results.get(i) for i in range(params.batch_size)]
        return image_batch
    else:
        await file_names._wait()
        file_name_results = file_names.wait()._output
        image_batch = PIL.Image.open(os.path.join(comfy_root_directory, "output", file_name_results["gifs"][0]["filename"]))
        return [image_batch]

async def _do_edit(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    with Workflow() as wf:
        workflow = model_definition.workflow(params)
        image_inputs = [LoadImage(filename)[0] for filename in [params.filename, params.filename2] if filename is not None]
        if len(image_inputs)> 1:
            image_input = ImageStitch(image_inputs[0], 'right', True, 0, 'white', image_inputs[1])
        else:
            image_input = image_inputs[0]
        image_input = workflow.resize_edit_image(image_input)
        workflow.create_img2img_latents(image_input)
        workflow.condition_prompts()
        workflow.edit_conditioning()
        workflow.sample()
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction, params.prompt))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch

async def _do_upscale(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    workflow = UpscaleWorkflow()
    workflow.load_image(params.filename)
    workflow.upscale(UPSCALE_DEFAULTS.model, 2.0)
    image = workflow.save("final_output")
    results = await image._wait()
    return await results.get(0)


async def _do_add_detail(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    with Workflow() as wf:
        workflow = model_definition.workflow(params)
        image_input = LoadImage(params.filename)[0]
        workflow.create_img2img_latents(image_input)
        workflow.condition_prompts()
        workflow.condition_for_detailing(params.detailing_controlnet, image_input)
        workflow.sample(use_ays=False)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction, params.prompt))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


async def _do_image_mashup(params: ImageWorkflow, model_definition: ModelDefinition, interaction):
    with Workflow() as wf:
        workflow = model_definition.workflow(params)
        image_inputs = [LoadImage(filename)[0] for filename in [params.filename, params.filename2] if filename is not None]
        workflow.create_latents()
        workflow.condition_prompts()
        workflow.unclip_encode(image_inputs)
        workflow.sample(use_ays=params.use_align_your_steps)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction, params.prompt))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


def process_prompt_with_llm(positive_prompt: str, seed: int, profile: str):
    from src.defaults import llm_prompt, llm_parameters

    prompt_text = llm_prompt + "\n" + positive_prompt
    _, prompt, _ = IFPromptMkr(
        input_prompt=prompt_text,
        engine=IFChatPrompt.engine.ollama,
        base_ip=llm_parameters["API_URL"],
        port=llm_parameters["API_PORT"],
        selected_model=llm_parameters["MODEL_NAME"],
        profile=profile,
        seed=seed,
        random=True,
    )
    return prompt


workflow_type_to_method = {
    WorkflowType.txt2img: _do_txt2img,
    WorkflowType.img2img: _do_img2img,
    WorkflowType.upscale: _do_upscale,
    WorkflowType.add_detail: _do_add_detail,
    WorkflowType.image_mashup: _do_image_mashup,
    WorkflowType.edit: _do_edit
}

user_queues = {}


def do_preview(task, node_id, image, interaction, prompt):
    if image is None:
        return
    try:
        filename = f"temp_preview_{task.prompt_id}.png"
        fp = os.path.join(comfy_root_directory, "output", filename)
        image.save(fp)
        if config["NSFW_DETECTION"]["NSFW_DETECTION_ENABLED"] == "True" and check_nsfw(fp, prompt) == True:
            return
        asyncio.run_coroutine_threadsafe(interaction.edit_original_response(attachments=[discord.File(fp, filename)]), loop)
    except Exception as e:
        print(e)


async def do_workflow(params: ImageWorkflow, model_definition: ModelDefinition, interaction: discord.Interaction):
    global user_queues, loop
    loop = asyncio.get_event_loop()
    user = interaction.user

    if user_queues.get(user.id) is not None and user_queues[user.id] >= int(config["BOT"]["MAX_QUEUE_PER_USER"]):
        await interaction.edit_original_response(
            content=f"{user.mention} `You have too many pending requests. Please wait for them to finish. Amount in queue: {user_queues[user.id]}`"
        )
        return

    if user_queues.get(user.id) is None or user_queues[user.id] < 0:
        user_queues[user.id] = 0

    user_queues[user.id] += 1

    queue.watch_display(False)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            loras = [Lora(lora, strength) for lora, strength in zip(params.loras, params.lora_strengths)] if params.loras else []

            extra_loras = get_loras_from_prompt(params.prompt)
            loras.extend([Lora(f"{lora[0]}.safetensors", lora[1]) for lora in extra_loras])

            if params.use_accelerator_lora and params.num_steps < 10:
                loras.append(Lora(params.accelerator_lora_name, 1.0))
                params.use_align_your_steps = False
            else:
                params.use_align_your_steps = True if params.model_type != ModelType.SD3 else False

            params.lora_dict = loras

            if params.use_llm is True:
                enhanced_prompt = process_prompt_with_llm(params.prompt, params.seed, params.llm_profile)
                prompt_result = await IFDisplayText(enhanced_prompt)
                params.prompt = params.prompt + ", BREAK \n" + prompt_result._output["string"][0]

            if params.style_prompt is not None and params.style_prompt not in params.prompt:
                params.prompt = params.style_prompt + "\n" + params.prompt
            if params.negative_style_prompt is not None and (params.negative_prompt is None or params.negative_style_prompt not in params.negative_prompt):
                params.negative_prompt = params.negative_style_prompt + "\n" + (params.negative_prompt or "")

            params.style_prompt = None
            params.negative_style_prompt = None

            result = await workflow_type_to_method[params.workflow_type](params, model_definition, interaction)

            user_queues[user.id] -= 1
            await interaction.edit_original_response(attachments=[])
            return result
        except Exception as e:
            print(f"Error during image generation: {e}")
            user_queues[user.id] -= 1
            retries += 1

    raise Exception("Failed to generate image")
