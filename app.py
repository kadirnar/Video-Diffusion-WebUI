import gradio as gr

from video_diffusion.damo.damo_text2_video import DamoText2VideoGenerator
from video_diffusion.inpaint_zoom.zoom_in_app import StableDiffusionZoomIn
from video_diffusion.inpaint_zoom.zoom_out_app import StableDiffusionZoomOut
from video_diffusion.stable_diffusion_video.stable_video_text2video import StableDiffusionText2VideoGenerator
from video_diffusion.tuneavideo.tuneavideo_text2video import TunaVideoText2VideoGenerator
from video_diffusion.zero_shot.zero_shot_text2video import ZeroShotText2VideoGenerator


def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Stable Diffusion Video"):
                    StableDiffusionText2VideoGenerator.app()
                with gr.Tab("Tune-a-Video"):
                    TunaVideoText2VideoGenerator.app()
                with gr.Tab("Stable Infinite Zoom"):
                    with gr.Tab("Zoom In"):
                        StableDiffusionZoomIn.app()
                    with gr.Tab("Zoom Out"):
                        StableDiffusionZoomOut.app()
                with gr.Tab("Damo Text2Video"):
                    DamoText2VideoGenerator.app()
                with gr.Tab("Zero Shot Text2Video"):
                    ZeroShotText2VideoGenerator.app()

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    diffusion_app()
