# DISPLAY NAMES OF NODE

SWITCH_SAMPLER_NODE_NAME = "SwitchSampler"
VAE_NODE_NAME = "VAEEncode (InpaintUpscale)"
PRINT_TIME_NODE_NAME = "PrintTime"
PROFILE_MODEL_NODE_NAME = "ProfileModel"
PROFILE_LATENT_NODE_NAME = "ProfileLatent"
PROFILE_NODE_NAME = "ProfileNode"
RELIGHT_NODE_NAME = "RelightNode"
RELIGHT_NODE_CUSTOM_NAME = "RelightNodeCustom"
CUSTOM_LOAD_IMAGE_NODE_NAME = "Custom Image as Mask Loader"
CUSTOM_LOAD_IMAGE_GENERAL_NODE_NAME = "Custom Image Loader (Tensor to PIL)"
CUSTOM_COMPOSITE_NODE_NAME = "Custom Composite (PIL to Tensor)"
IMAGE_RESIZE_CUSTOM = "Custom Image Resize (PIL to PIL)"
CUSTOM_FINAL = "Custom Final"


# OPEN AI

OPEN_AI_BASE_URL = "https://api.openai.com/v1/chat/completions"
OPEN_AI_IMAGE_MODEL = "gpt-4o"

# PROMPT NEEDS

JSON_SCHEMA_GPT_TEXT = {
    "type": "object",
    "properties": {
        "light_yaw": {
            "type": "integer"
        },
        "light_pitch": {
            "type": "integer"
        }
    },
    "required": [
        "light_yaw",
        "light_pitch"
    ]
}


SAMPLE_JSON = {
    "light_yaw": 60,
    "light_pitch": 40,
    "specular_power": 6,
    "ambient_light": 1.00,
    "normal_diffuse_strength": 0.45,
    "depth_diffuse_strength": 0.00,
    "total_gain": 1.00
}

BACKGROUND_COLOR_FOR_COMPOSITE = (0, 0, 0, 0)
MAX_RESOLUTION = 16384