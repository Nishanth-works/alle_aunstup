HAR_CASCADES_PATH = "haarcascade_frontalface_default.xml"
FACE_SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 5
MIN_SIZE = (30, 30)

LOWER_SKIN_BOUNDARIES = [0, 133, 77]
UPPER_SKIN_BOUNDARIES = [255, 173, 127]


# VIT_MATTE
VIT_MATTE_MODEL_NAME = "vitmatte-s"
VIT_MATTE_MODEL_CHECKPOINT = "/checkpoints/ViTMatte_S_Com.pth"

# HSV
THRESHOLD = 60

# DIRECTORY
DIRECTORY_TO_SAVE_VIT_MATTE = "./vit-matte-results"
MODEL_DIR = "./checkpoints/ViTMatte_S_Com.pth"
DIRECTORY_TO_SAVE_MODIFIED_MATTE = "./modified_matte"
DIRECTORY_TO_SAVE_IMAGE_OVERLAY = "./image_overlay"

# EMBEDDING MODEL
EMBEDDING_MODEL_NAME = "imagenet"
POLLING_EMBEDDING = "avg"
