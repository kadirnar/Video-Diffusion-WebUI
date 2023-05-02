import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import gradio as gr


class DamoText2VideoGenerator:
    def __init__(self):
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload()
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
    ):
        pipe = self.load_model()
        video = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
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
                    dano_text2video_prompt = gr.Textbox(lines=1, placeholder="Prompt")
                    dano_text2video_negative_prompt = gr.Textbox(lines=1, placeholder="Negative Prompt")
                    with gr.Row():
                        with gr.Column():
                            dano_text2video_num_inference_steps = gr.Number(value=50, label="Number of Inference Steps")
                            dano_text2video_guidance_scale = gr.Number(value=7.5, label="Guidance Scale")
                        with gr.Row():
                            with gr.Column():

                                dano_text2video_height = gr.Slider(
                                    minimum=1,
                                    maximum=1280,
                                    value=512,
                                    label="Height",
                                )
                                dano_text2video_width = gr.Slider(
                                    minimum=1,
                                    maximum=1280,
                                    value=512,
                                    label="Width",
                                )
                                dano_text2video_num_frames = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    value=10,
                                    label="Number of Frames",
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
            ],
            outputs=dano_output,
        )
