import comfy.samplers
import comfy.sample
import datetime
import comfy.model_management
import latent_preview
import math
import inspect
import folder_paths
import os
from .constants import SAMPLE_JSON, JSON_SCHEMA_GPT_TEXT, BACKGROUND_COLOR_FOR_COMPOSITE, MAX_RESOLUTION
from .prompts import IMAGE_RELIGHT_PROMPT
from .utils import create_openai_image_query, relight_image_using_shade_light, create_open_ai_query_text, \
    vectorize_euler_value, convert_greyscale_tensor_to_transparent, segment_image_using_mask, pillow, pil2tensor, \
    tensor2pil
import torch
import numpy as np
import hashlib
from PIL import Image, ImageOps


class AbstractAssistantNode(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


abstract_assistance_node = AbstractAssistantNode("*")


class PrintTime:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"node": (abstract_assistance_node,)}
        }

    RETURN_TYPES = ()

    FUNCTION = "print_time"

    OUTPUT_NODE = True

    CATEGORY = "display"

    def print_time(self, node):
        executed_at = datetime.datetime.now()
        print(executed_at)


class ProfileModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",),
                         "text": ("STRING", {"multiline": True})}
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "print_time"
    CATEGORY = "profiling"

    def print_time(self, model, text=""):
        executed_at = datetime.datetime.now()
        print(text, executed_at)
        return (model,)


class ProfileLatent:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"latent": ("LATENT",),
                         "text": ("STRING", {"multiline": True})}
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "print_time"
    CATEGORY = "profiling"

    def print_time(self, latent, text=""):
        executed_at = datetime.datetime.now()
        print(text, executed_at)
        return (latent,)


class ProfileNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"any_input": (abstract_assistance_node,),
                         "text": ("STRING", {"multiline": True})}
        }

    RETURN_TYPES = (abstract_assistance_node,)
    FUNCTION = "print_time"
    CATEGORY = "profiling"

    def print_time(self, any_input, text=""):
        executed_at = datetime.datetime.now()
        print(text, executed_at)
        return (any_input,)


class SwitchSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_1": ("MODEL",),
                     "model_2": ("MODEL",),
                     "switch_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model_1, model_2, switch_step, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent, denoise=1.0):

        disable_noise = False
        start_step = None
        last_step = None
        force_full_denoise = False
        latent_image = latent["samples"]
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_callback = latent_preview.prepare_callback(model_1, steps)

        def model_switch():
            comfy.model_management.load_model_gpu(model_2)
            model_1.model = model_2.model
            model_1.patches = model_2.patches
            return

        def callback(step, x0, x, total_steps):
            if step == switch_step:
                model_switch()
            preview_callback(step, x0, x, total_steps)

        disable_pbar = False
        samples = comfy.sample.sample(model_1, noise, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent_image,
                                      denoise=denoise, disable_noise=disable_noise, start_step=start_step,
                                      last_step=last_step,
                                      force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback,
                                      disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return (out,)


class VAEEncodeForInpaintUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE",), "vae": ("VAE",), "mask": ("MASK",),
                             "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}), }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/inpaint"

    def encode(self, vae, pixels, mask, grow_mask_by=6):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        # grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)
        t = vae.encode(pixels)

        return ({"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())},)


class RelightNode:
    @classmethod
    def INPUT_TYPES(cls):
        from .enums import RELIGHT_NODE_INPUT_TYPES
        return RELIGHT_NODE_INPUT_TYPES

    RETURN_TYPES = ("IMAGE", "STRING",)
    FUNCTION = "image_relight"
    CATEGORY = "custom"

    def image_relight(self, original_image, depth_map, normal_map, text=""):
        prompt_light_direction = ("Specify the Light direction like right top to left bottom , left to right etc"
                                  + IMAGE_RELIGHT_PROMPT["GPT_LIGHT_DIRECTION_DETECTION_PROMPT"])
        response_for_image_information = create_openai_image_query(
            original_image,
            prompt_light_direction
        )
        print(response_for_image_information)
        CODE_LIGHT = inspect.getsource(vectorize_euler_value)
        CODE = inspect.getsource(relight_image_using_shade_light)
        if response_for_image_information["success"]:
            value_prompt_mapping = {
                "image_details": response_for_image_information["data"],
                "code_light_direction": CODE_LIGHT,
                "code_lighting": CODE,
                "json_format": JSON_SCHEMA_GPT_TEXT
            }
        else:
            value_prompt_mapping = {
                "image_details": "Its a fashion model image.",
                "code_light_direction": CODE_LIGHT,
                "code_lighting": CODE,
                "json_format": JSON_SCHEMA_GPT_TEXT
            }

        value_of_prompt = IMAGE_RELIGHT_PROMPT["GPT4_PROMPT_TEST"]
        for key, value in value_prompt_mapping.items():
            value_of_prompt = value_of_prompt.replace("{" + key + "}", str(value))
        final_prompt = text + value_of_prompt
        gpt_response = create_open_ai_query_text(final_prompt)
        print("gpt4 text in use")
        print(gpt_response)
        if gpt_response["success"]:
            try:
                response = gpt_response["data"]
                response_light_yaw = response["light_yaw"]
                response_light_pitch = response["light_pitch"]
                response_specular_power = SAMPLE_JSON["specular_power"]
                response_ambient_light = SAMPLE_JSON["ambient_light"]
                response_normal_diffuse_strength = SAMPLE_JSON["normal_diffuse_strength"]
                response_depth_diffuse_strength = SAMPLE_JSON["depth_diffuse_strength"]
                response_total_gain = SAMPLE_JSON["total_gain"]
                relight_image = relight_image_using_shade_light(original_image, normal_map,
                                                                depth_map, response_light_yaw,
                                                                response_light_pitch,
                                                                response_specular_power,
                                                                response_ambient_light,
                                                                response_normal_diffuse_strength,
                                                                float(response_depth_diffuse_strength),
                                                                response_total_gain)
                print(relight_image)
                if relight_image["success"]:
                    return relight_image["output_tensor"], "Successfully relighted the image using GPT"
                else:
                    return "", relight_image["error"]
            except:
                print("Entered on default light mode! OPEN AI crashed in making json")
                relight_image = relight_image_using_shade_light(original_image, normal_map,
                                                                depth_map, SAMPLE_JSON["light_yaw"],
                                                                SAMPLE_JSON["light_pitch"],
                                                                SAMPLE_JSON["specular_power"],
                                                                SAMPLE_JSON["ambient_light"],
                                                                SAMPLE_JSON["normal_diffuse_strength"],
                                                                SAMPLE_JSON["depth_diffuse_strength"],
                                                                SAMPLE_JSON["total_gain"])
                if relight_image["success"]:
                    return relight_image["output_tensor"], "Successfully relighted the image using default mode"
                else:
                    return "", relight_image["error"]
        else:
            print(f'Entered on default light mode! OPEN AI crashed :{gpt_response["error"]}')
            relight_image = relight_image_using_shade_light(original_image, normal_map,
                                                            depth_map, SAMPLE_JSON["light_yaw"],
                                                            SAMPLE_JSON["light_pitch"],
                                                            SAMPLE_JSON["specular_power"],
                                                            SAMPLE_JSON["ambient_light"],
                                                            SAMPLE_JSON["normal_diffuse_strength"],
                                                            SAMPLE_JSON["depth_diffuse_strength"],
                                                            SAMPLE_JSON["total_gain"])
            if relight_image["success"]:
                return relight_image["output_tensor"], ("Successfully relighted the image using default mode as Open ai"
                                                        "crashed")
            else:
                return "", relight_image["error"]


class RelightNodeCustom:
    @classmethod
    def INPUT_TYPES(cls):
        from .enums import RELIGHT_NODE_CUSTOM_INPUT_TYPES
        return RELIGHT_NODE_CUSTOM_INPUT_TYPES

    RETURN_TYPES = ("IMAGE", "STRING",)
    FUNCTION = "image_relight_custom"
    CATEGORY = "custom"

    def image_relight_custom(self, original_image, depth_map, normal_map, light_yaw, light_pitch, specular_power,
                             ambient_light, normal_diffuse_strength, depth_diffuse_strength, total_gain, text=""):
        relight_image = relight_image_using_shade_light(original_image, normal_map,
                                                        depth_map, light_yaw,
                                                        light_pitch,
                                                        specular_power,
                                                        ambient_light,
                                                        normal_diffuse_strength,
                                                        depth_diffuse_strength,
                                                        total_gain)
        if relight_image["success"]:
            return relight_image["output_tensor"], "Successfully relighted the image using GPT"
        else:
            return "", relight_image["error"]

class CustomLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("PIL Image")
    FUNCTION = "load_image_custom"

    def load_image_custom(self, image):
        from PIL import Image
        image_path = folder_paths.get_annotated_filepath(image)
        img = pillow(Image.open, image_path)
        return (img, )


class CustomLoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (s._color_channels, ),
                     "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01}),}
                }
    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"

    def load_image(self, image, channel, rescale_factor=2):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        if rescale_factor:
            new_size = (int(i.width * rescale_factor), int(i.height * rescale_factor))
            i = i.resize(new_size)
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            # if c == 'A':
            #     mask = 1. - mask
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class ImageResizeCustom:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["rescale", "resize"],),
                "supersample": (["true", "false"],),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resize_width": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                "resize_height": ("INT", {"default": 1536, "min": 1, "max": 48000, "step": 1}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("PIL Image",)
    FUNCTION = "image_rescale"
    CATEGORY = "custom"
    def image_rescale(self, image: Image.Image, mode="rescale", supersample='true', resampling="lanczos",
                      rescale_factor=2, resize_width=1024, resize_height=1024):
        resized_image = self.apply_resize_image(image, mode, supersample, rescale_factor, resize_width, resize_height,
                                                resampling)
        return (resized_image,)

    def apply_resize_image(self, image: Image.Image, mode='rescale', supersample='true', factor: int = 2,
                           width: int = 1024, height: int = 1024, resample='bicubic') -> Image.Image:
        current_width, current_height = image.size
        if mode == 'rescale':
            new_width, new_height = int(current_width * factor), int(current_height * factor)
        else:
            new_width = width if width % 8 == 0 else width + (8 - width % 8)
            new_height = height if height % 8 == 0 else height + (8 - height % 8)
        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }
        if supersample == 'true':
            image = image.resize((new_width * 8, new_height * 8), resample=resample_filters[resample])
        resized_image = image.resize((new_width, new_height), resample=resample_filters[resample])

        return resized_image

class CustomComposite:
    @classmethod
    def INPUT_TYPES(s):
        from .enums import CUSTOM_COMPOSITE
        return CUSTOM_COMPOSITE

    CATEGORY = "custom"

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Final Composite", "Cutout")
    FUNCTION = "custom_composite"

    def custom_composite(self, background, image_mask, original_image, x, y, resize_source=False):
        from PIL import Image, ImageFilter, ImageOps
        import torch
        import numpy as np
        import random

        image_mask = image_mask.convert("RGBA")
        original_image = original_image.convert("RGBA")
        mask_alpha = image_mask.split()[-1]
        segmented_image = Image.new("RGBA", original_image.size, BACKGROUND_COLOR_FOR_COMPOSITE)

        actual_image_array = np.array(original_image)
        mask_alpha_array = np.array(mask_alpha)
        actual_image_tensor = torch.from_numpy(actual_image_array).float() / 255.0
        mask_alpha_tensor = torch.from_numpy(mask_alpha_array).float() / 255.0

        for c in range(3):
            actual_image_tensor[:, :, c] *= mask_alpha_tensor

        masked_image_array = (actual_image_tensor * 255).byte().numpy()
        masked_image = Image.fromarray(masked_image_array, "RGBA")
        segmented_image.paste(masked_image, (0, 0), mask_alpha)
        # cutout

        background = background.convert("RGBA")
        print(background.mode)

        if resize_source:
            background = background.resize(segmented_image.size)

        overlay = segmented_image.convert("RGBA")
        overlay = overlay.copy()
        alpha = overlay.split()[-1]
        alpha = alpha.point(lambda p: p * 1.0)
        overlay.putalpha(alpha)
        overlay_width, overlay_height = overlay.size
        background_width, background_height = background.size

        x = max(-overlay_width, min(x, background_width))
        y = max(-overlay_height, min(y, background_height))

        # Calculate the position and cropping of the overlay image
        left = max(0, x)
        top = max(0, y)
        right = min(background_width, x + overlay_width)
        bottom = min(background_height, y + overlay_height)

        overlay_cropped = overlay.crop((left - x, top - y, right - x, bottom - y))
        background_cropped = background.crop((left, top, right, bottom))

        result = Image.alpha_composite(background_cropped, overlay_cropped)
        background.paste(result, (left, top))
        result = background.convert("RGBA")
        # result = ImageOps.crop(result, border=1)
        return (pil2tensor(result), pil2tensor(segmented_image))

class CustomFinal:
    @classmethod
    def INPUT_TYPES(s):
        from .enums import CUSTOM_COMPOSITE_IMAGE
        return CUSTOM_COMPOSITE_IMAGE

    CATEGORY = "custom"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Final Composite")
    FUNCTION = "custom_composite"

    def custom_composite(self, background, image_mask, original_image):
        from PIL import Image, ImageFilter
        original_image = original_image.convert("RGBA")  # Original RGB image
        alpha_matte = image_mask.convert("RGBA")  # Alpha matte image
        background_image = background.convert("RGBA")  # Background RGB image

        # Ensure the images are the same size
        original_image = original_image.resize(alpha_matte.size)
        background_image = background_image.resize(alpha_matte.size)

        # Create a new image for the cutout using the alpha matte as the alpha channel
        cutout = Image.new("RGBA", original_image.size)
        cutout.paste(original_image, (0, 0), mask=alpha_matte)

        # Overlay the cutout on the background image
        combined_image = Image.alpha_composite(background_image, cutout)
        # base_image = background.convert("RGBA")
        # cutout = original_image.convert("RGBA")
        # alpha_only = cutout.split()[-3]
        # composite = Image.composite(cutout, base_image, alpha_only)
        # background = background.convert("RGB")
        # original_image = original_image.convert("RGB")
        # image_mask = image_mask.convert("L")
        # mask_resized = image_mask.filter(ImageFilter.MinFilter(3))
        # result = Image.composite(original_image, background, image_mask)
        return (pil2tensor(combined_image), )