import os

import gradio as gr
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from video_diffusion.inpaint_zoom.utils.zoom_in_utils import dummy, image_grid, shrink_and_paste_on_blank, write_video

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


stable_paint_model_list = ["stabilityai/stable-diffusion-2-inpainting", "runwayml/stable-diffusion-inpainting"]

stable_paint_prompt_list = [
    "children running in the forest , sunny, bright, by studio ghibli painting, superior quality, masterpiece,  traditional Japanese colors, by Grzegorz Rutkowski, concept art",
    "A beautiful landscape of a mountain range with a lake in the foreground",
]

stable_paint_negative_prompt_list = [
    "lurry, bad art, blurred, text, watermark",
]


class StableDiffusionZoomIn:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_id):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to("cuda")
            self.pipe.safety_checker = dummy
            self.pipe.enable_attention_slicing()
            self.pipe.enable_xformers_memory_efficient_attention()
            self.g_cuda = torch.Generator(device="cuda")

        return self.pipe

    def generate_video(
        self,
        model_id,
        prompt,
        negative_prompt,
        guidance_scale,
        num_inference_steps,
    ):
        pipe = self.load_model(model_id)

        num_init_images = 2
        seed = 42
        height = 512
        width = height

        current_image = Image.new(mode="RGBA", size=(height, width))
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")
        current_image = current_image.convert("RGB")

        init_images = pipe(
            prompt=[prompt] * num_init_images,
            negative_prompt=[negative_prompt] * num_init_images,
            image=current_image,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=self.g_cuda.manual_seed(seed),
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
        )[0]

        image_grid(init_images, rows=1, cols=num_init_images)

        init_image_selected = 1  # @param
        if num_init_images == 1:
            init_image_selected = 0
        else:
            init_image_selected = init_image_selected - 1

        num_outpainting_steps = 20  # @param
        mask_width = 128  # @param
        num_interpol_frames = 30  # @param

        current_image = init_images[init_image_selected]
        all_frames = []
        all_frames.append(current_image)

        for i in range(num_outpainting_steps):
            print("Generating image: " + str(i + 1) + " / " + str(num_outpainting_steps))

            prev_image_fix = current_image

            prev_image = shrink_and_paste_on_blank(current_image, mask_width)

            current_image = prev_image

            # create mask (black image with white mask_width width edges)
            mask_image = np.array(current_image)[:, :, 3]
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")

            # inpainting step
            current_image = current_image.convert("RGB")
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                # this can make the whole thing deterministic but the output less exciting
                # generator = g_cuda.manual_seed(seed),
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
            )[0]
            current_image = images[0]
            current_image.paste(prev_image, mask=prev_image)

            # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
            for j in range(num_interpol_frames - 1):
                interpol_image = current_image
                interpol_width = round(
                    (1 - (1 - 2 * mask_width / height) ** (1 - (j + 1) / num_interpol_frames)) * height / 2
                )
                interpol_image = interpol_image.crop(
                    (interpol_width, interpol_width, width - interpol_width, height - interpol_width)
                )

                interpol_image = interpol_image.resize((height, width))

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = round((1 - (height - 2 * mask_width) / (height - 2 * interpol_width)) / 2 * height)
                prev_image_fix_crop = shrink_and_paste_on_blank(prev_image_fix, interpol_width2)
                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

                all_frames.append(interpol_image)

            all_frames.append(current_image)

        video_file_name = "infinite_zoom_out"
        fps = 30
        save_path = video_file_name + ".mp4"
        write_video(save_path, all_frames, fps)
        return save_path

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    text2image_in_model_path = gr.Dropdown(
                        choices=stable_paint_model_list, value=stable_paint_model_list[0], label="Text-Image Model Id"
                    )

                    text2image_in_prompt = gr.Textbox(lines=2, value=stable_paint_prompt_list[0], label="Prompt")

                    text2image_in_negative_prompt = gr.Textbox(
                        lines=1, value=stable_paint_negative_prompt_list[0], label="Negative Prompt"
                    )

                    with gr.Row():
                        with gr.Column():
                            text2image_in_guidance_scale = gr.Slider(
                                minimum=0.1, maximum=15, step=0.1, value=7.5, label="Guidance Scale"
                            )

                            text2image_in_num_inference_step = gr.Slider(
                                minimum=1, maximum=100, step=1, value=50, label="Num Inference Step"
                            )

                    text2image_in_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Video(label="Output")

            text2image_in_predict.click(
                fn=StableDiffusionZoomIn().generate_video,
                inputs=[
                    text2image_in_model_path,
                    text2image_in_prompt,
                    text2image_in_negative_prompt,
                    text2image_in_guidance_scale,
                    text2image_in_num_inference_step,
                ],
                outputs=output_image,
            )
