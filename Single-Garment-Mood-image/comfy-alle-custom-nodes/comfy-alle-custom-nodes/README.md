# comfy-caimera-nodes

## SwitchSampler
Similar to KSampler but takes 2 models and an additiona `switch step` as input.
It starts denoising with `mdoel_1` It uses comfy callback function to check if the current denoising step is equal to the `switch step`, if yes it replaces model_1 with model_2 and uses that for the remaining steps. Note that it doesn't take separate inputs for conditioning moedel_1 and model_2. If different conditioning is required then ContitioningCombine with proper start and end timesteps can be provided.

## Profile Node
Takes input of any type (defined by datatype AnyType). It has the same output as the input, it just prints current system time. If the input node was cached, then it is also cached and does not print anything. ProfileLatent, and ProfileModel are input type specific nodes that are now redundant because of ProfileNode and can be removed. `PrintTime` is a terminal node which can accept any input and prints system time.

## VAEEncodeForInpaintUpscale
This is a VAEEncode node that can be used for inpainting. It's difference from Comfy's VAEEncodeForInpaint is that the later changes the masked area to empty image whereas it keeps the masked area same as before and adds the mask to the latent dictionary in the same way as the former node. It can be used with KSampler to inpaint with low denoising strength in a masked area. Inpainting with lower denoising strength is not possible when using the VAEEncodeForInpaint node, as low denoising strength for empty image doesn't produce expected output.

## Relighting Node:
Relighting Node, lits up an exsisting image using its normal and depth map. 
![alt text](https://raw.githubusercontent.com/bahaal-tech/comfy-caimera-nodes/main/diagram.png?token=GHSAT0AAAAAACRZRS644JJLPDTUVW4IBPCWZRURY2Q)

# Git patches

## upscaler
Takes mask as input. Keeps the masked area constant and denoises in the non-masked region.

## switches
Custome switches for ksampler, face_restore nodes. The switch returns the relevant inputs as outputs when turned off, executes the expected function when turned ON.
