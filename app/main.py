import gradio as gr
from app.model import predict_caption

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🖼️ Image Caption Generator")
    gr.Markdown("BLIP Model (Vision-Language Transformer)")

    with gr.Row():
        
        # -------- Input Section --------
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Image")

            with gr.Accordion("Advanced Parameters", open=False):
                max_len = gr.Slider(
                    minimum=10, maximum=50, value=30, step=1,
                    label="Max Caption Length"
                )
                beams = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Beam Search Size"
                )

            submit_btn = gr.Button("Generate Caption", variant="primary")

        # -------- Output Section --------
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Caption",
                lines=3,
                placeholder="Your caption will appear here..."
            )

    # ---------------------------
    # Button Action
    # ---------------------------
    submit_btn.click(
        fn=predict_caption,
        inputs=[input_img, max_len, beams],  # removed temperature
        outputs=output_text
    )

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    demo.launch()