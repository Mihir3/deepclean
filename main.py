import argparse
import os
from PIL import Image
from lang_sam import LangSAM
#set system path
import sys
sys.path.append("/u/anua2/deepclean")
from lora.lora_diffusion.lora import patch_pipe
from lora.lora_diffusion.lora import tune_lora_scale
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
import torch

import numpy as np

def save_mask(mask, output_path):
    """
    Saves the mask as a PNG image.
    Args:
        mask: The mask to save.
        output_path (str): Path to save the mask image.
    """
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save(output_path)

def detect_and_segment(src_image_path: str, remove_objects: list, output_dir: str):
    """
    Detects and segments objects in the image based on the removal prompt.
    Args:
        src_image_path (str): Path to the source image.
        remove_objects (list): List of objects to remove.
    """
    src_image = Image.open(src_image_path)
    lang_sam_model = LangSAM()
    print(remove_objects)
    results = lang_sam_model.predict([src_image], remove_objects)
    if not results or results[0]["masks"] is None:
        raise ValueError("No masks found for the given objects.")
    mask = results[0]["masks"][0]
    
    save_mask(mask, os.path.join(output_dir, "mask.png"))

def guided_inpainting(src_image_path: str, inpainting_prompt: str, output_dir: str):
    """
    Performs guided inpainting using the given mask and prompt.
    Args:
        src_image_path (str): Path to the source image.
        mask: Mask from segmentation step.
        inpainting_prompt (str): Prompt describing replacement content.
        output_dir (str): Directory to save the output.
    """
    mask_path = os.path.join(output_dir, "mask.png")
    base_image = Image.open(src_image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")
    model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # disables loading it entirely
    ).to("cuda")
    torch.manual_seed(6)
    
    patch_pipe(
        pipe,
        "../test_exps/test-coco_text_extended_inpainting/final_lora.safetensors",
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )


    tune_lora_scale(pipe.unet, 0.5)
    tune_lora_scale(pipe.text_encoder, 0.5)
    ft_inpainted_image = pipe(prompt=inpainting_prompt, image=base_image, mask_image=mask_image, num_inference_steps=100, guidance_scale=7).images[0]
    ft_inpainted_image.save(os.path.join(output_dir, "inpainted_image.png"))
    print(f"Inpainted image saved to {os.path.join(output_dir, 'inpainted_image.png')}")


    
def main():
    parser = argparse.ArgumentParser(description="Text-guided object removal and inpainting pipeline")
    parser.add_argument("--src_image_path", type=str, required=True, help="Path to the source image")
    parser.add_argument("--remove_object", type=str, required=True, help="Object(s) to remove, separated by dots")
    parser.add_argument("--inpainting_prompt", type=str, required=True, help="Prompt describing what to replace the object with")
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save the results")

    args = parser.parse_args()

    # Split multiple objects using '.' delimiter
    remove_objects = [obj.strip() for obj in args.remove_object.split('.') if obj.strip()]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Detect and segment
    detect_and_segment(args.src_image_path, remove_objects, args.output_dir)

    # Step 2: Guided inpainting
    guided_inpainting(args.src_image_path, args.inpainting_prompt, args.output_dir)

if __name__ == "__main__":
    main()
