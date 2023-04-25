import torch
from stable_diffusion_videos import StableDiffusionWalkPipeline

class StableDiffusionText2VideoGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(
        self,
        model_path,
    ):
        if self.pipe is None:
            self.pipe = StableDiffusionWalkPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                revision="fp16",
            )

        self.pipe.to(torch_device="cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def generate_video(
        self,
        model_path: str,
        prompts: str,
        negative_prompt: str,
        num_interpolation_steps: int,
        guidance_scale: int,
        num_inference_step: int,
        height: int,
        width: int,
        seeds: list,
    ):
        pipe = self.load_model(model_path=model_path)
        output_video = pipe.walk(
            prompts=prompts,
            num_interpolation_steps=num_interpolation_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_step,
            negative_prompt=negative_prompt,
            seeds=seeds,
        )

        return output_video
