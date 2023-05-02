import os

import gradio as gr
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from video_diffusion.inpaint_zoom.utils.zoom_out_utils import (
    dummy,
    preprocess_image,
    preprocess_mask_image,
    write_video,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


stable_paint_model_list = ["stabilityai/stable-diffusion-2-inpainting", "runwayml/stable-diffusion-inpainting"]

stable_paint_prompt_list = [
    "children running in the forest , sunny, bright, by studio ghibli painting, superior quality, masterpiece,  traditional Japanese colors, by Grzegorz Rutkowski, concept art",
    "A beautiful landscape of a mountain range with a lake in the foreground",
]

stable_paint_negative_prompt_list = [
    "lurry, bad art, blurred, text, watermark",
]


class StableDiffusionZoomOut:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_id):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe.set_use_memory_efficient_attention_xformers(True)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to("cuda")
            self.pipe.safety_checker = dummy
            self.g_cuda = torch.Generator(device="cuda")

        return self.pipe

    def generate_video(
        self,
        model_id,
        prompt,
        negative_prompt,
        guidance_scale,
        num_inference_steps,
        num_frames,
        step_size,
    ):
        pipe = self.load_model(model_id)

        new_image = Image.new(mode="RGBA", size=(512, 512))
        current_image, mask_image = preprocess_mask_image(new_image)

        current_image = pipe(
            prompt=[prompt],
            negative_prompt=[negative_prompt],
            image=current_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        all_frames = []
        all_frames.append(current_image)

        for i in range(num_frames):
            prev_image = preprocess_image(current_image, step_size, 512)
            current_image = prev_image
            current_image, mask_image = preprocess_mask_image(current_image)
            current_image = pipe(
                prompt=[prompt],
                negative_prompt=[negative_prompt],
                image=current_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
            ).images[0]
            current_image.paste(prev_image, mask=prev_image)
            all_frames.append(current_image)

        save_path = "output.mp4"
        write_video(save_path, all_frames, fps=30)
        return save_path

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2image_out_model_path = gr.Dropdown(
                        choices=stable_paint_model_list, value=stable_paint_model_list[0], label="Text-Image Model Id"
                    )

                    text2image_out_prompt = gr.Textbox(lines=2, value=stable_paint_prompt_list[0], label="Prompt")

                    text2image_out_negative_prompt = gr.Textbox(
                        lines=1, value=stable_paint_negative_prompt_list[0], label="Negative Prompt"
                    )

                    with gr.Row():
                        with gr.Column():
                            text2image_out_guidance_scale = gr.Slider(
                                minimum=0.1, maximum=15, step=0.1, value=7.5, label="Guidance Scale"
                            )

                            text2image_out_num_inference_step = gr.Slider(
                                minimum=1, maximum=100, step=1, value=50, label="Num Inference Step"
                            )
                        with gr.Row():
                            with gr.Column():
                                text2image_out_step_size = gr.Slider(
                                    minimum=1, maximum=100, step=1, value=10, label="Step Size"
                                )

                                text2image_out_num_frames = gr.Slider(
                                    minimum=1, maximum=100, step=1, value=10, label="Frames"
                                )

                    text2image_out_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Video(label="Output Video")

            text2image_out_predict.click(
                fn=StableDiffusionZoomOut().generate_video,
                inputs=[
                    text2image_out_model_path,
                    text2image_out_prompt,
                    text2image_out_negative_prompt,
                    text2image_out_guidance_scale,
                    text2image_out_num_inference_step,
                    text2image_out_step_size,
                    text2image_out_num_frames,
                ],
                outputs=output_image,
            )
