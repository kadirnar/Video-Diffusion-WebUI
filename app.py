import gradio as gr

from video_diffusion import  StableDiffusionText2VideoGenerator

def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Video"):
                    StableDiffusionText2VideoGenerator.app()
                
                
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    diffusion_app()