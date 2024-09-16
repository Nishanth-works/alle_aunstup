from .nodes import SwitchSampler, VAEEncodeForInpaintUpscale, PrintTime, ProfileModel, \
    ProfileLatent, ProfileNode, RelightNode, RelightNodeCustom, CustomLoadImageMask, CustomLoadImage, CustomComposite, \
    ImageResizeCustom, CustomFinal
from .constants import SWITCH_SAMPLER_NODE_NAME, PROFILE_NODE_NAME, PROFILE_MODEL_NODE_NAME, RELIGHT_NODE_NAME, \
    VAE_NODE_NAME, PROFILE_LATENT_NODE_NAME, PRINT_TIME_NODE_NAME, RELIGHT_NODE_CUSTOM_NAME, \
    CUSTOM_LOAD_IMAGE_NODE_NAME, MAX_RESOLUTION, CUSTOM_COMPOSITE_NODE_NAME, IMAGE_RESIZE_CUSTOM, CUSTOM_FINAL, \
    CUSTOM_LOAD_IMAGE_GENERAL_NODE_NAME

NODE_CLASS_MAPPINGS = {
    "SwitchSampler": SwitchSampler,
    "VAEEncodeForInpaintUpscale": VAEEncodeForInpaintUpscale,
    "PrintTime": PrintTime,
    "ProfileModel": ProfileModel,
    "ProfileLatent": ProfileLatent,
    "ProfileNode": ProfileNode,
    "RelightNode": RelightNode,
    "RelightNodeCustom": RelightNodeCustom,
    "CustomLoadImageMask": CustomLoadImageMask,
    "CustomLoadImage": CustomLoadImage,
    "CustomComposite": CustomComposite,
    "ImageResizeCustom": ImageResizeCustom,
    "CustomFinal": CustomFinal
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwitchSampler": SWITCH_SAMPLER_NODE_NAME,
    "VAEEncodeForInpaintUpscale": VAE_NODE_NAME,
    "PrintTime": PRINT_TIME_NODE_NAME,
    "ProfileModel": PROFILE_MODEL_NODE_NAME,
    "ProfileLatent": PROFILE_LATENT_NODE_NAME,
    "ProfileNode": PROFILE_NODE_NAME,
    "RelightNode": RELIGHT_NODE_NAME,
    "RelightNodeCustom": RELIGHT_NODE_CUSTOM_NAME,
    "CustomLoadImageMask": CUSTOM_LOAD_IMAGE_NODE_NAME,
    "CustomLoadImage": CUSTOM_LOAD_IMAGE_GENERAL_NODE_NAME,
    "CustomComposite": CUSTOM_COMPOSITE_NODE_NAME,
    "ImageResizeCustom": IMAGE_RESIZE_CUSTOM,
    "CustomFinal": CUSTOM_FINAL
}

# Relight Node

RELIGHT_NODE_INPUT_TYPES = {
    "required": {
        "original_image": ("IMAGE",),
        "depth_map": ("IMAGE",),
        "normal_map": ("IMAGE",),
    },
    "optional": {
        "text": ("STRING", {"multiline": True, "default": ""})}
}

RELIGHT_NODE_CUSTOM_INPUT_TYPES = {
    "required": {
        "original_image": ("IMAGE",),
        "depth_map": ("IMAGE",),
        "normal_map": ("IMAGE",),
        "light_yaw": ("FLOAT", {"default": 45, "min": -180, "max": 180, "step": 1}),
        "light_pitch": ("FLOAT", {"default": 30, "min": -90, "max": 90, "step": 1}),
        "specular_power": ("FLOAT", {"default": 32, "min": 1, "max": 200, "step": 1}),
        "ambient_light": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.01}),
        "normal_diffuse_strength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
        "depth_diffuse_strength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
        "total_gain": ("FLOAT", {"default": 1.00, "min": 0, "max": 2.0, "step": 0.01}),
    },
    "optional": {
        "text": ("STRING", {"multiline": True, "default": ""})}
}

CUSTOM_NODE_BLEND = {
    "required": {
        "image": ("IMAGE",),
        "original_image": ("IMAGE",)
    }}

CUSTOM_COMPOSITE = {
    "required": {
        "background": ("IMAGE",),
        "image_mask": ("IMAGE",),
        "original_image": ("IMAGE",),
        "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        "resize_source": ("BOOLEAN", {"default": False}),
    }
}

CUSTOM_COMPOSITE_IMAGE = {
    "required": {
        "background": ("IMAGE",),
        "image_mask": ("IMAGE",),
        "original_image": ("IMAGE",)
    }
}
