import torch
import os
import cv2
from constants import VIT_MATTE_MODEL_NAME, THRESHOLD, DIRECTORY_TO_SAVE_VIT_MATTE, \
    DIRECTORY_TO_SAVE_MODIFIED_MATTE, MODEL_DIR
from utils import (model_initializer, alpha_matte_inference_from_vision_transformer,
                   selective_search_and_remove_skin_tone, convert_greyscale_image_to_transparent)


class SkinSegmentVitMatte:
    def __init__(self):
        """
        loads the vit matte model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_matte_model = model_initializer(model=VIT_MATTE_MODEL_NAME, checkpoint=MODEL_DIR,
                                                 device=self.device)

    def generate_modified_matted_results(self, input_image, trimap_image):
        """
        :param input_image: this is the original image whose matte is to be generated type: str path
        :param trimap_image: trimap is the image generated using our trimap generation code type: str path
        :return: dict of generated images including matte type : dict
        """
        cutout_image_from_vit_matting = alpha_matte_inference_from_vision_transformer(self.vit_matte_model, input_image,
                                                                                      trimap_image,
                                                                                      DIRECTORY_TO_SAVE_VIT_MATTE)
        if not cutout_image_from_vit_matting["success"]:
            error_message = f"Vit matting failed due to: {cutout_image_from_vit_matting}"
            return {"success": False, "error": error_message}

        modified_matte = selective_search_and_remove_skin_tone(input_image,
                                                               cutout_image_from_vit_matting["vit_matte_output"],
                                                               THRESHOLD, DIRECTORY_TO_SAVE_MODIFIED_MATTE)

        if not modified_matte["success"]:
            error_message = (f"skin removal from the matte failed due to :{modified_matte['error']}, so we will be using"
                             f"default vit matte output path.")
            modified_matte_path_need_to_be_passed = cutout_image_from_vit_matting["vit_matte_output"]
        else:
            modified_matte_path_need_to_be_passed = modified_matte["output"]
        modified_matte_image = cv2.imread(modified_matte_path_need_to_be_passed)
        kernel_for_modified_matte = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        final_image = cv2.morphologyEx(modified_matte_image, cv2.MORPH_OPEN, kernel_for_modified_matte, iterations=3)
        dir_final = os.path.join(DIRECTORY_TO_SAVE_MODIFIED_MATTE, "final_matte_image.png")
        cv2.imwrite(dir_final, final_image)
        grey_scale_final_path = os.path.join(DIRECTORY_TO_SAVE_MODIFIED_MATTE, "edge_less_final_matte.png")
        convert_greyscale_image_to_transparent(dir_final, grey_scale_final_path)
        return {"success": True, "vit_matte_path": cutout_image_from_vit_matting["vit_matte_output"],
                "non_converted_final_mask": dir_final,
                "vit_matte_cutout_image": cutout_image_from_vit_matting["cutout_output"],
                "skin_cut_output": modified_matte_path_need_to_be_passed,
                "edge_less_no_mask": grey_scale_final_path,
                "modified_matte_path": grey_scale_final_path
                }
