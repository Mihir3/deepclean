from PIL import Image
from lora_diffusion.lora import patch_pipe
from lora_diffusion.lora import tune_lora_scale


base_image = Image.open("/u/mpamnani/lora/coco_text_extended/9_src.png")
mask_image = Image.open("/u/mpamnani/lora/coco_text_extended/9_mask.png")

# base_image.show()
# mask_image.show()

from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,  # disables loading it entirely
).to("cuda")

# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Disable safety filter
# def dummy_checker(images, **kwargs): return images, False
# pipe.safety_checker = dummy_checker


base_prompt = "write START on the sign board"
torch.manual_seed(6)
image = pipe(prompt=base_prompt, image=base_image, mask_image=mask_image, num_inference_steps=100, guidance_scale=7).images[0]
  # nice. diffusers are cool.
image.save("base-coco-inpaint.jpg")

ft_prompt = "write <s1> START on the sign board"
patch_pipe(
    pipe,
    "../test_exps/test-coco_text_extended_inpainting/final_lora.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)


tune_lora_scale(pipe.unet, 0.5)
tune_lora_scale(pipe.text_encoder, 0.5)
ft_0_8_inpainted_image = pipe(prompt=ft_prompt, image=base_image, mask_image=mask_image, num_inference_steps=100, guidance_scale=7).images[0]
ft_0_8_inpainted_image.save("0.5-ft-coco-inpaint.jpg")
print(ft_0_8_inpainted_image)
