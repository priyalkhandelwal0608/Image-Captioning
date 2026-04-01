#  Image Captioning using BLIP

An AI-powered web application that generates natural language captions for images using the **BLIP (Bootstrapped Language Image Pretraining)** model and an interactive **Gradio UI**.

---

##  Features

  * Upload any image and generate captions instantly
  * Uses state-of-the-art **Vision-Language Transformer (BLIP)**
  * Adjustable parameters:

  * Max caption length
  * Beam search size
  * GPU acceleration support (CUDA)
  * Clean and interactive Gradio interface

---

## Project Structure

```
Image-Captioning/
│── app/
│   ├── model.py        # BLIP model + caption generation logic
│   ├── ui.py           # Gradio UI
│
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

##  Model Used

* **Model**: `Salesforce/blip-image-captioning-base`
* Architecture:

  * Vision Encoder (ViT)
  * Text Decoder (Transformer)

---

## ⚙️ Installation


pip install -r requirements.txt
```

---

##  Run the Application

```bash
python app/ui.py
```

Then open:

```
http://127.0.0.1:7860/
```

---

##  Usage

1. Upload an image
2. (Optional) Adjust advanced parameters
3. Click **"Generate Caption"**
4. View AI-generated caption

---





If you like this project, give it a ⭐ on GitHub!
