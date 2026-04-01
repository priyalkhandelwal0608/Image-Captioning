import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# Model Setup
# ---------------------------
model_id = "Salesforce/blip-image-captioning-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
model.eval()


# ---------------------------
# Caption Function
# ---------------------------
def predict_caption(image, max_length=30, num_beams=5):
    if image is None:
        return "Please upload an image."

    # Ensure correct format
    image = image.convert("RGB")
    image = image.resize((384, 384))

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate caption
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model.generate(
                **inputs,
                max_length=int(max_length),
                num_beams=int(num_beams),
                do_sample=False,              # deterministic → better accuracy
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )

    # Decode output
    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    # Clean formatting
    caption = caption.capitalize()
    if not caption.endswith("."):
        caption += "."

    return caption