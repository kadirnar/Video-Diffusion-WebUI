import gradio as gr
import imageio
import torch

from video_diffusion.tuneavideo.util import save_videos_grid
from video_diffusion.utils.model_list import stable_model_list
from diffusers import TextToVideoZeroPipeline


class ZeroShotText2VideoGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_id):
        if self.pipe is None:
            self.pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
            self.pipe.to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_attention_slicing()
        return self.pipe

    def generate_video(
        self,
        prompt,
        negative_prompt,
        model_id,
        height,
        width,
        video_length,
        num_inference_steps,
        guidance_scale,
        fps,
        t0,
        t1,
        motion_field_strength_x,
        motion_field_strength_y,
    ):
        breakpoint()
        pipe = self.load_model(model_id)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            t0=t0,
            t1=t1,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
        ).images
        
        result = [(r * 255).astype("uint8") for r in result]
        save_videos_grid(videos=result, path="output.gif", fps=fps)
        return "output.gif"

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    zero_shot_text2video_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )
                    zero_shot_text2video_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )
                    zero_shot_text2video_model_id = gr.Dropdown(
                        choices=stable_model_list,
                        label="Stable Model List",
                        value=stable_model_list[0],
                    )
                    with gr.Row():
                        with gr.Column():
                            zero_shot_text2video_num_inference_steps = gr.Slider(
                                label="Number of Inference Steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=30,
                            )
                            zero_shot_text2video_guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1,
                                maximum=15,
                                step=1,
                                value=7.5,
                            )
                            zero_shot_text2video_video_length = gr.Slider(
                                label="Video Length",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=10,
                            )
                            zero_shot_text2video_t0 = gr.Slider(
                                label="T0",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=0,
                            )
                            zero_shot_text2video_motion_field_strength_x = gr.Slider(
                                label="Motion Field Strength X",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=12,
                            )
                        with gr.Row():
                            with gr.Column():
                                zero_shot_text2video_height = gr.Slider(
                                    label="Height",
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                )
                                zero_shot_text2video_width = gr.Slider(
                                    label="Width",
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                )
                                zero_shot_text2video_fps = gr.Slider(
                                    label="Fps",
                                    minimum=1,
                                    maximum=60,
                                    step=1,
                                    value=10,
                                )
                                zero_shot_text2video_t1 = gr.Slider(
                                    label="T1",
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    value=10,
                                )
                                zero_shot_text2video_motion_field_strength_y = gr.Slider(
                                    label="Motion Field Strength Y",
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    value=12,
                                )
                    zero_shot_text2video_button = gr.Button(value="Generator")

                with gr.Column():
                    zero_shot_text2video_output = gr.Video(label="Output")

            zero_shot_text2video_button.click(
                fn=ZeroShotText2VideoGenerator().generate_video,
                inputs=[
                    zero_shot_text2video_prompt,
                    zero_shot_text2video_negative_prompt,
                    zero_shot_text2video_model_id,
                    zero_shot_text2video_height,
                    zero_shot_text2video_width,
                    zero_shot_text2video_video_length,
                    zero_shot_text2video_num_inference_steps,
                    zero_shot_text2video_guidance_scale,
                    zero_shot_text2video_fps,
                    zero_shot_text2video_t0,
                    zero_shot_text2video_t1,
                    zero_shot_text2video_motion_field_strength_x,
                    zero_shot_text2video_motion_field_strength_y,
                ],
                outputs=zero_shot_text2video_output,
            )
