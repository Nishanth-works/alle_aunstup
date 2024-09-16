# Alle Projects
This repository contains all projects workflows, codes and training configs for Alle.

## Project - Single Garment Mood Image:
1.) First setup ComfyUI/

2.) Download the mentioned models, in Inpainting , Upscaler and expand workflow and place it in specific models/ folders, if any doubt please ask

3.) Next put comfy-alle-custom-nodes in custom_nodes/ folder 

4.) Then apply ```git-commit-patch```

5.) Now you are ready to use the ComfyUI

6.) Next deploy the image-matting code somewhere, I will suggest take an A10 server in Replicate and deploy there.

7.) Pass it the segmented garment mask, it will return High res Alpha mask of that garment

8.) Check Garment mask and product image both have same W x H or not (obviously they should if not do it)

9.) Now you are ready to use the workflow, put the necessary inputs in necessary places in the workflow, add face, pose and background reference if you want to use the IPA

## Project - Influencer Lora:
1.) Training config is provided in the repo, we are using Ai tool kit

2.) Use network volume anustup-alle-loras in Runpod, everything is present there

3.) Just download 20-30 hires (>1K) images of the influencer 

4.) Run the GPT caption script provided in the server 

5.) Hit the training

6.) Once you get the lora, turn on Flux text workflow to test with some prompts

7.) To improve quality use latent injection

## Body Shape and Size:
1.) This is SDXL lora training

2.) Download different body shape and size images, caption them like esk pear shaped extra plus

3.) Preset data is already present 

4.) Train the lora and test 
