import torch
from stable_diffusion_videos import StableDiffusionWalkPipeline
from video_diffusion.utils.model_list import stable_model_list
import gradio as gr

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
        first_seeds: int,
        second_seeds: int,
    ):
        seeds = [first_seeds, second_seeds]
        prompts = [int(first_prompts), int(second_prompts)]
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
    
    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2video_first_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Enter text first prompt here",
                        show_label=False,
                    )
                    text2video_second_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Enter text second prompt here",
                        show_label=False,
                    )
                    text2video_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Enter negative text prompt here",
                        show_label=False,
                    )
                
                    with gr.Row():
                        with gr.Column():
                            text2video_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                label="Model",
                                value=stable_model_list[0],
                            )
                                

                            text2video_guidance_scale = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=8.5,
                                label="Guidance scale",
                            )
                            text2video_num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Number of Inference Steps",
                            )
                        with gr.Row():
                            with gr.Column():
                                text2video_num_interpolation_steps = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=3,
                                    value=10,
                                    label="Number of Interpolation Steps",
                                )
                                text2video_height = gr.Slider(
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                    value=512,
                                    label="Height",
                                )
                                text2video_width = gr.Slider(
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                    value=512,
                                    label="Width",
                                )

                                text2video_first_seeds = gr.Number(
                                    value=0,
                                    label="Seed",
                                )
                                text2video_second_seeds = gr.Number(
                                    value=0,
                                    label="Seed",
                                )
                    text2video_generate = gr.Button(value="Generator")
            
                with gr.Column():
                    text2video_output = gr.Video(value=None, label="Output video").style(grid=(1, 2), height=200)

            text2video_generate.click(
                fn=StableDiffusionText2VideoGenerator().generate_video,
                inputs=[
                    text2video_model_path,
                    text2video_first_prompt,
                    text2video_second_prompt,
                    text2video_negative_prompt,
                    text2video_num_interpolation_steps,
                    text2video_guidance_scale,
                    text2video_num_inference_steps,
                    text2video_height,
                    text2video_width,
                    text2video_first_seeds,
                    text2video_second_seeds,
                ],
                outputs=text2video_output
            )
