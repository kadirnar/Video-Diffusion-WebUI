# https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero

import logging
from typing import Optional

import gradio as gr
import imageio
import torch
from diffusers import TextToVideoZeroPipeline, TextToVideoZeroSDXLPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stable_model_list = ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"]


class ZeroShotText2VideoGenerator:

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
        if model_type == "sd15":
            pipe = TextToVideoZeroPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id, torch_dtype=torch.float16).to(self.device)

        elif model_type == "sdxl":
            pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True).to(self.device)

        # memory optimization
        pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        pipe.enable_vae_slicing()

        logging.info(f"Model loaded: {model_id}")

        self.pipe = pipe

    def generate_video(
        self,
        prompt,
        negative_prompt,
        model_id,
        height,
        width,
        video_length,
        guidance_scale,
        fps,
        t0,
        t1,
        motion_field_strength_x,
        motion_field_strength_y,
        model_type="sd15",
        save_path: Optional[str] = "output.mp4",
    ):
        pipe = self.load_model(model_id, model_type=model_type)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            video_length=video_length,
            guidance_scale=guidance_scale,
            t0=t0,
            t1=t1,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
        ).images

        result = [(r * 255).astype("uint8") for r in result]
        imageio.mimsave(save_path, result, fps=fps)
        return save_path

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
                    zero_shot_text2video_model_type = gr.Dropdown(
                        choices=["sd15", "sdxl"],
                        label="Model Type",
                        value="sd15",
                    )

                    with gr.Row():
                        with gr.Column():
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
                                label="Timestep T0",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=44,
                            )
                            zero_shot_text2video_motion_field_strength_x = gr.Slider(
                                label="Motion Field Strength X",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=12,
                            )
                            zero_shot_text2video_fps = gr.Slider(
                                label="Fps",
                                minimum=1,
                                maximum=60,
                                step=1,
                                value=10,
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
                                zero_shot_text2video_t1 = gr.Slider(
                                    label="Timestep T1",
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    value=47,
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
                    zero_shot_text2video_guidance_scale,
                    zero_shot_text2video_fps,
                    zero_shot_text2video_t0,
                    zero_shot_text2video_t1,
                    zero_shot_text2video_motion_field_strength_x,
                    zero_shot_text2video_motion_field_strength_y,
                    zero_shot_text2video_model_type,
                ],
                outputs=zero_shot_text2video_output,
            )
