# https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video

import logging
from typing import Optional

import gradio as gr
import imageio
import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

text2video_model_list = [
    "ali-vilab/text-to-video-ms-1.7b", "cerspense/zeroscope_v2_576w", "strangeman3107/animov-512x"
]


class ModelscopeText2VideoGenerator:

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = None
        self.device = None

        if self.pipe is None:
            self.load_model(model_id)
        else:
            logging.info("Model already loaded.")

        self.set_device()

    def set_device(self):
        """Sets the device to be used for inference based on availability."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Using device: {self.device}")

    def load_model(self, model_id, model_type="sd15"):
        logging.info(f"Loading model: {model_id}")

        pipe = TextToVideoSDPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id, torch_dtype=torch.float16,
            variant="fp16").to(self.device)

        # memory optimization
        pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        pipe.enable_vae_slicing()

        logging.info(f"Model loaded: {model_id}")

        self.pipe = pipe

    def generate_video(
        self,
        model_id,
        prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        num_frames,
        fps,
        save_path: Optional[str] = "output.mp4",
    ):
        pipe = self.load_model(model_id)

        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
        ).frames[0]

        video_path = export_to_video(result, save_path, fps=fps)

        return video_path

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2video_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )
                    text2video_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )
                    text2video_model_id = gr.Dropdown(
                        choices=text2video_model_list,
                        label="Model ID",
                        value="ali-vilab/text-to-video-ms-1.7b",
                    )
                    with gr.Row():
                        with gr.Column():
                            text2video_guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=0.5,
                            )
                            text2video_video_num_frames = gr.Slider(
                                label="Video Length",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=20,
                            )
                            text2video_fps = gr.Slider(
                                label="FPS",
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=10,
                            )
                        with gr.Row():
                            with gr.Column():
                                text2video_height = gr.Slider(
                                    label="Height",
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                )
                                text2video_width = gr.Slider(
                                    label="Width",
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                )
                                text2video_num_inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=20,
                                )

                    text2video_button = gr.Button(value="Generator")

                with gr.Column():
                    text2video_output = gr.Video(label="Output")

            text2video_button.click(
                fn=ModelscopeText2VideoGenerator().generate_video,
                inputs=[
                    text2video_model_id,
                    text2video_prompt,
                    text2video_height,
                    text2video_width,
                    text2video_num_inference_steps,
                    text2video_guidance_scale,
                    text2video_negative_prompt,
                    text2video_video_num_frames,
                    text2video_fps,
                ],
                outputs=text2video_output,
            )
