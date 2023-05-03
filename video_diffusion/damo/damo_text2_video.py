import gradio as gr
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

from video_diffusion.utils.scheduler_list import diff_scheduler_list, get_scheduler_list


class DamoText2VideoGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, scheduler):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
            )
            self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        return self.pipe

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: int,
        height: int,
        width: int,
        scheduler: str,
    ):
        pipe = self.load_model(scheduler=scheduler)
        video = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_frames=int(num_frames),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames

        video_path = export_to_video(video)
        return video_path

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    dano_text2video_prompt = gr.Textbox(lines=1, placeholder="Prompt", show_label=False)
                    dano_text2video_negative_prompt = gr.Textbox(
                        lines=1, placeholder="Negative Prompt", show_label=False
                    )
                    with gr.Row():
                        with gr.Column():
                            dano_text2video_num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Inference Steps",
                            )
                            dano_text2video_guidance_scale = gr.Slider(
                                minimum=1,
                                maximum=15,
                                value=7,
                                step=1,
                                label="Guidance Scale",
                            )
                            dano_text2video_num_frames = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=16,
                                step=1,
                                label="Number of Frames",
                            )
                        with gr.Row():
                            with gr.Column():
                                dano_text2video_height = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    value=512,
                                    step=32,
                                    label="Height",
                                )
                                dano_text2video_width = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    value=512,
                                    step=32,
                                    label="Width",
                                )
                                damo_text2video_scheduler = gr.Dropdown(
                                    choices=diff_scheduler_list,
                                    label="Scheduler",
                                    value=diff_scheduler_list[6],
                                )
                    dano_text2video_generate = gr.Button(value="Generator")
                with gr.Column():
                    dano_output = gr.Video(label="Output")

        dano_text2video_generate.click(
            fn=DamoText2VideoGenerator().generate_video,
            inputs=[
                dano_text2video_prompt,
                dano_text2video_negative_prompt,
                dano_text2video_num_frames,
                dano_text2video_num_inference_steps,
                dano_text2video_guidance_scale,
                dano_text2video_height,
                dano_text2video_width,
                damo_text2video_scheduler,
            ],
            outputs=dano_output,
        )
