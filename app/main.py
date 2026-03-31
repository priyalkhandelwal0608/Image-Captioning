import gradio as gr
from app.model import predict_caption

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Image Caption Generator")
    gr.Markdown("Encoder-Decoder model using ViT + GPT-2")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Image")

            with gr.Accordion("Advanced Parameters", open=False):
                max_len = gr.Slider(10, 50, value=20, label="Max Caption Length")
                beams = gr.Slider(1, 10, value=5, step=1, label="Beam Search Size")
                temp = gr.Slider(0.0, 1.5, value=1.0, label="Temperature")

            submit_btn = gr.Button("Generate Caption", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Caption")

    submit_btn.click(
        fn=predict_caption,
        inputs=[input_img, max_len, beams, temp],
        outputs=output_text
    )

demo.launch()