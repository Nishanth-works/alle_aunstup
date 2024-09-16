import torch

from .constants import OPEN_AI_BASE_URL, OPEN_AI_IMAGE_MODEL
import os
import requests
import numpy as np
import json
from PIL import Image
from dotenv import load_dotenv
import base64
import PIL
from torch import Tensor
from PIL import ImageFile, UnidentifiedImageError

load_dotenv()

OPEN_AI_API_KEY = os.getenv('OPENAI_API_KEY')


def tensor2pil(image: Tensor) -> PIL.Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2base64(image: PIL.Image.Image) -> str:
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_openai_image_query(input_image: torch.Tensor, prompt):
    b64image = pil2base64(tensor2pil(input_image))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPEN_AI_API_KEY}"
    }
    payload = {
        "model": OPEN_AI_IMAGE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text",
             "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64image}"}}]}],
        "response_format": {"type": "json_object"}}
    response = requests.post(OPEN_AI_BASE_URL, headers=headers, json=payload)
    try:
        json_response = response.json()
        selected_json_response = json_response["choices"][0]["message"]["content"]
        final_response = json.loads(selected_json_response)
        return {"success": True, "data": final_response}
    except:
        return {"success": False, "error": f"Open AI crashed, find full log here:{response.json()}"}


def vectorize_euler_value(rotation_around_vertical_axis, rotation_around_side_to_side_axis):
    rotation_radius = np.radians(rotation_around_vertical_axis)
    side_rotation_radius = np.radians(rotation_around_side_to_side_axis)

    cos_of_side_rotation = np.cos(side_rotation_radius)
    sin_of_side_rotation = np.sin(side_rotation_radius)
    cos_of_vertical_rotation = np.cos(rotation_radius)
    sin_of_vertical_rotation = np.sin(rotation_radius)

    direction_of_shading_light = np.array([
        sin_of_vertical_rotation * cos_of_side_rotation,
        sin_of_side_rotation,
        cos_of_side_rotation * cos_of_vertical_rotation
    ])
    return torch.from_numpy(direction_of_shading_light).float()


def relight_image_using_shade_light(original_image_tensor, normal_map_tensor, depth_map_tensor, light_yaw,
                                    light_pitch, specular_power, ambient_light,
                                    normal_diffuse_strength, depth_diffuse_strength, total_gain):
    original_image_modified_tensor = original_image_tensor.permute(0, 3, 1, 2)
    normal_map_modified_tensor = normal_map_tensor.permute(0, 3, 1, 2) * 2.0 - 1.0
    depth_map_modified_tensor = depth_map_tensor.permute(0, 3, 1, 2)
    normalized_surface_normal_tensor = torch.nn.functional.normalize(normal_map_modified_tensor, dim=1)

    light_direction = vectorize_euler_value(light_yaw, light_pitch)
    light_direction = light_direction.view(1, 3, 1, 1)
    camera_direction = vectorize_euler_value(0, 0)
    camera_direction = camera_direction.view(1, 3, 1, 1)
    try:
        diffuse_light = torch.sum(normalized_surface_normal_tensor * light_direction, dim=1, keepdim=True)
        diffuse_light = torch.clamp(diffuse_light, 0, 1)
        half_vector = torch.nn.functional.normalize(light_direction + camera_direction, dim=1)
        depth_light = torch.sum(normalized_surface_normal_tensor * half_vector, dim=1, keepdim=True)
        depth_light = torch.pow(torch.clamp(depth_light, 0, 1), specular_power)
        output_tensor = (original_image_modified_tensor * (
                ambient_light + diffuse_light * normal_diffuse_strength) + depth_map_modified_tensor *
                         depth_light * depth_diffuse_strength) * total_gain
        output_tensor = output_tensor.permute(0, 2, 3, 1)
        return {"success": True, "output_tensor": output_tensor}
    except Exception as e:
        return {"success": False, "error": f"Relighting failed due to : {e}"}


def create_open_ai_query_text(input_query, system_message=None, model_engine="gpt-4o",
                              functions=None, function_call=None):
    openai_url = f"https://api.openai.com/v1/chat/completions"
    headers = {'Authorization': f'Bearer {OPEN_AI_API_KEY}', 'Content-Type': 'application/json'}
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": input_query})
    payload = {
        'model': model_engine,
        'messages': messages,
        'response_format': {"type": "json_object"}
    }
    if functions:
        payload['functions'] = functions
        payload['function_call'] = function_call
    response = requests.post(openai_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200 and 'choices' in response.json():
        if functions:
            content_text = response.json()['choices'][0]['message']['function_call']['arguments'].strip()
        else:
            content_text = response.json()['choices'][0]['message']['content'].strip()
        final_response = json.loads(content_text)
        return {"success": True, "data": final_response, "response_json": response.json()}
    return {"success": False, "error": response.text}

def convert_greyscale_tensor_to_transparent(input_tensor):
    alpha_channel = input_tensor.clone()
    color_channel = (alpha_channel > 0).float() * 255
    alpha_image = torch.stack([color_channel, color_channel, color_channel, alpha_channel], dim=-1)
    return alpha_image


def segment_image_using_mask(image_tensor, mask_tensor):
    background_color = [0.0, 0.0, 0.0, 0.0]
    background_tensor = torch.zeros_like(image_tensor)
    for c in range(4):
        background_tensor[:, :, c] = background_color[c]
    for c in range(3):
        image_tensor[:, :, c] *= mask_tensor
    mask_alpha_tensor_expanded = mask_tensor.unsqueeze(-1)
    masked_image_tensor = image_tensor * mask_alpha_tensor_expanded
    background_image_tensor = background_tensor * (1 - mask_alpha_tensor_expanded)
    segmented_image_tensor = masked_image_tensor + background_image_tensor
    return segmented_image_tensor

def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return x

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
