import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Better model than ViT-GPT2
model_id = "Salesforce/blip-image-captioning-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + processor
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)


def predict_caption(image, max_length, num_beams, temperature):
    if image is None:
        return "Please upload an image."

    # Resize for consistency (improves results)
    image = image.resize((384, 384))

    # Convert image → tensor
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption (NO randomness → better accuracy)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=int(max_length),
            num_beams=int(num_beams),
            do_sample=False,              # IMPORTANT: accuracy ↑
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption.capitalize()