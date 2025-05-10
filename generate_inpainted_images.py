import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import argparse
from PIL import Image
import random
import numpy as np

def main(args):
    image_path = args.image
    mask_path = args.mask
    prompt = args.prompt
    output_dir = args.output_dir
    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)

    num_inference_steps = 30
    guidance_scale = 20.0
    strength = 0.99
    padding_mask_crop = 2
    RES = (640, 640)
    TARGET = (1024, 1024)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    image = load_image(image_path).resize(TARGET)
    mask = load_image(mask_path).resize(TARGET)

    # Load pipeline
    print(f"Loading model diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ).to(device)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
        height=TARGET[1],
        width=TARGET[0],
        original_size=TARGET,
        target_size=TARGET,
        padding_mask_crop=padding_mask_crop
    ).images[0]

    result = result.resize(RES)
    mask_resized = mask.resize(RES)

    result.save(os.path.join(output_dir, "generated.png"))
    mask_resized.save(os.path.join(output_dir, "mask.png"))

    print("Saved generated image and mask.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the source image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the mask image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args)
