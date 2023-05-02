import gradio as gr
import numpy as np
import torch

from video_diffusion.stable_diffusion_video.stable_diffusion_pipeline import StableDiffusionWalkPipeline
from video_diffusion.utils.model_list import stable_model_list


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
        first_prompts: str,
        second_prompts: str,
        negative_prompt: str,
        num_interpolation_steps: int,
        guidance_scale: int,
        num_inference_step: int,
        height: int,
        width: int,
        upsample: bool,
        fps=int,
    ):
        first_seed = np.random.randint(0, 100000)
        second_seed = np.random.randint(0, 100000)
        seeds = [first_seed, second_seed]
        prompts = [first_prompts, second_prompts]
        pipe = self.load_model(model_path=model_path)

        output_video = pipe.walk(
            prompts=prompts,
            num_interpolation_steps=int(num_interpolation_steps),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_step,
            negative_prompt=negative_prompt,
            seeds=seeds,
            upsample=upsample,
            fps=fps,
        )

        return output_video

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    stable_text2video_first_prompt = gr.Textbox(
                        lines=1,
                        placeholder="First Prompt",
                        show_label=False,
                    )
                    stable_text2video_second_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Second Prompt",
                        show_label=False,
                    )
                    stable_text2video_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt ",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            stable_text2video_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                label="Stable Model List",
                                value=stable_model_list[0],
                            )
                            stable_text2video_guidance_scale = gr.Slider(
                                minimum=0,
                                maximum=15,
                                step=1,
                                value=8.5,
                                label="Guidance Scale",
                            )
                            stable_text2video_num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=30,
                                label="Number of Inference Steps",
                            )
                            stable_text2video_fps = gr.Slider(
                                minimum=1,
                                maximum=60,
                                step=1,
                                value=10,
                                label="Fps",
                            )
                        with gr.Row():
                            with gr.Column():
                                stable_text2video_num_interpolation_steps = gr.Number(
                                    value=10,
                                    label="Number of Interpolation Steps",
                                )
                                stable_text2video_height = gr.Slider(
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                    value=512,
                                    label="Height",
                                )
                                stable_text2video_width = gr.Slider(
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                    value=512,
                                    label="Width",
                                )
                                stable_text2video_upsample = gr.Checkbox(
                                    label="Upsample",
                                    default=False,
                                )

                    text2video_generate = gr.Button(value="Generator")

                with gr.Column():
                    text2video_output = gr.Video(label="Output")

            text2video_generate.click(
                fn=StableDiffusionText2VideoGenerator().generate_video,
                inputs=[
                    stable_text2video_model_path,
                    stable_text2video_first_prompt,
                    stable_text2video_second_prompt,
                    stable_text2video_negative_prompt,
                    stable_text2video_num_interpolation_steps,
                    stable_text2video_guidance_scale,
                    stable_text2video_num_inference_steps,
                    stable_text2video_height,
                    stable_text2video_width,
                    stable_text2video_upsample,
                    stable_text2video_fps,
                ],
                outputs=text2video_output,
            )
