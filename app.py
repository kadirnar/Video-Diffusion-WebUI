import gradio as gr

from video_diffusion.text2video.modelscope_pipeline import ModelscopeText2VideoGenerator
from video_diffusion.text2video.zero_pipeline import ZeroShotText2VideoGenerator


def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Zero Shot Text2Video"):
                    ZeroShotText2VideoGenerator.app()
                with gr.Tab("Text2Video"):
                    ModelscopeText2VideoGenerator.app()

    app.launch(debug=True)


if __name__ == "__main__":
    diffusion_app()
