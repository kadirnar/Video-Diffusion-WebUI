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
        seeds: int,
        upsample: bool,
    ):
        #seeds = [int(first_seeds), int(second_seeds)]
        prompts = [first_prompts, second_prompts]
        pipe = self.load_model(model_path=model_path)
        seeds = seeds.replace("[", "").replace("]", "").split(",")
        
        output_video = pipe.walk(
            prompts=prompts,
            num_interpolation_steps=num_interpolation_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_step,
            negative_prompt=negative_prompt,
            seeds=seeds,
            upsample=upsample,
        )

        return output_video
    
    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2video_first_prompt = gr.Textbox(
                        lines=1,
                        placeholder="First Prompt",
                        show_label=False,
                    )
                    text2video_second_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Second Prompt",
                        show_label=False,
                    )
                    text2video_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt ",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            text2video_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                label="Stable Model List",
                                value=stable_model_list[0],
                            )
                            text2video_guidance_scale = gr.Slider(
                                minimum=0,
                                maximum=15,
                                step=1,
                                value=8.5,
                                label="Guidance Scale",
                            )
                            text2video_num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Number of Inference Steps",
                            )
                            text2video_seeds = gr.Textbox(
                                lines=1,
                                placeholder="Seeds: [42, 224]",
                                show_label=False,
                            )
                        with gr.Row():
                            with gr.Column():

                                text2video_num_interpolation_steps = gr.Number(
                                    value=3,
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
                                text2video_upsample = gr.Checkbox(
                                    label="Upsample",
                                    default=False,
                                )


                    text2video_generate = gr.Button(value="Generator")
            
                with gr.Column():
                    text2video_output = gr.Video(value=None, label="Output video")

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
                    text2video_seeds,
                    text2video_upsample,
                ],
                outputs=text2video_output
            )
