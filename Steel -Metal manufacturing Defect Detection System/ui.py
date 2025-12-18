import gradio as gr
import requests
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

def gradio_predict(image):
    if image is None:
        return "No image uploaded"

    try:
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Send image to FastAPI
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()

        result = response.json()

        # Safely get class name and confidence
        class_name = result.get('class') or result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0) * 100

        return f"Predicted Defect: {class_name}\nConfidence: {confidence:.2f}%"

    except requests.exceptions.RequestException as e:
        return f"API request error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Steel Surface Image"),
    outputs="text",
    title="Steel & Metal Surface Defect Detection"
)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)