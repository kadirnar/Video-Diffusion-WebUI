import gradio as gr
import torch

from video_diffusion.tuneavideo.models.unet import UNet3DConditionModel
from video_diffusion.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from video_diffusion.tuneavideo.util import save_videos_grid
from video_diffusion.utils.model_list import stable_model_list

video_diffusion_model_list = [
    "Tune-A-Video-library/a-man-is-surfing",
    "Tune-A-Video-library/mo-di-bear-guitar",
    "Tune-A-Video-library/redshift-man-skiing",
]


class TunaVideoText2VideoGenerator:
    def __init__(self):
        self.pipe = None
        self.unet = None

    def load_model(self, video_diffusion_model_list, stable_model_list):
        if self.pipe is None:
            if self.unet is None:
                self.unet = UNet3DConditionModel.from_pretrained(
                    video_diffusion_model_list, subfolder="unet", torch_dtype=torch.float16
                ).to("cuda")

            self.pipe = TuneAVideoPipeline.from_pretrained(
                stable_model_list, unet=self.unet, torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_vae_slicing()

        return self.pipe

    def generate_video(
        self,
        video_diffusion_model: str,
        stable_model_list: str,
        prompt: str,
        negative_prompt: str,
        video_length: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: int,
        fps: int,
    ):
        pipe = self.load_model(video_diffusion_model, stable_model_list)
        video = pipe(
            prompt,
            negative_prompt=negative_prompt,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).videos

        output_path = save_videos_grid(videos=video, save_path='output', path=f"{prompt}.gif")
        return output_path

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    tunevideo_video_diffusion_model_list = gr.Dropdown(
                        choices=video_diffusion_model_list,
                        label="Video Diffusion Model",
                        value=video_diffusion_model_list[0],
                    )
                    tunevideo_stable_model_list = gr.Dropdown(
                        choices=stable_model_list,
                        label="Stable Model List",
                        value=stable_model_list[0],
                    )
                    with gr.Row():
                        with gr.Column():
                            tunevideo_prompt = gr.Textbox(
                                lines=1,
                                placeholder="Prompt",
                                show_label=False,
                            )
                            tunevideo_video_length = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=10,
                                label="Video Length",
                            )

                            tunevideo_num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Steps",
                            )
                            tunevideo_fps = gr.Slider(
                                minimum=1,
                                maximum=60,
                                step=1,
                                value=30,
                                label="Fps",
                            )
                        with gr.Row():
                            with gr.Column():
                                tunevideo_negative_prompt = gr.Textbox(
                                    lines=1,
                                    placeholder="Negative Prompt",
                                    show_label=False,
                                )
                                tunevideo_guidance_scale = gr.Slider(
                                    minimum=1,
                                    maximum=15,
                                    step=1,
                                    value=7.5,
                                    label="Guidance Scale",
                                )
                                tunevideo_height = gr.Slider(
                                    minimum=1,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Height",
                                )
                                tunevideo_width = gr.Slider(
                                    minimum=1,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Width",
                                )
                    tunevideo_generate = gr.Button(value="Generator")

                with gr.Column():
                    tunevideo_output = gr.Video(label="Output")

            tunevideo_generate.click(
                fn=TunaVideoText2VideoGenerator().generate_video,
                inputs=[
                    tunevideo_video_diffusion_model_list,
                    tunevideo_stable_model_list,
                    tunevideo_prompt,
                    tunevideo_negative_prompt,
                    tunevideo_video_length,
                    tunevideo_height,
                    tunevideo_width,
                    tunevideo_num_inference_steps,
                    tunevideo_guidance_scale,
                    tunevideo_fps,
                ],
                outputs=tunevideo_output,
            )
