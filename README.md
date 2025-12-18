---

# Steel & Metal Manufacturing Defect Detection System

A CNN-based defect detection system using ResNet18, designed for a fabrication shop called **Chamal Steel Fabricators**.  
This is a **reproducible version** of the project, intended for demonstration and easy setup.
![1]([https://raw.githubusercontent.com/JanithDoesCode/Steel-Metal-Manufacturing-Defect-Detection-System/blob/main/screenshots/1.png](https://github.com/JanithDoesCode/Steel-Metal-Manufacturing-Defect-Detection-System/blob/main/screenshots/1.png))

---

## Project Overview

This project detects surface defects in steel and metal sheets. It uses:

- **ResNet18** CNN model for classification
- **FastAPI** backend for serving predictions via API
- **Gradio UI** for an easy-to-use interface

The system is designed for small to medium fabrication shops to quickly test defect detection on their steel surfaces.

---

## Folder Structure

Steel-Metal-Defect-Detection/ │ ├─ app.py                    # FastAPI backend ├─ predict.py                # Model loading + prediction logic ├─ inference.py              # Optional, separate inference file ├─ ui.py                     # Gradio interface ├─ defect_model_resnet18.pth # Pre-trained model (~50MB) ├─ requirements.txt          # Python dependencies ├─ README.md                 # This file ├─ screenshots/              # Optional, add UI and prediction screenshots

**Note:**  
The `ML Workflow` folder is included for  and Model training code(most important part) reference, but is **not part of the reproducible setup**. It contains experimental notebooks, intermediate datasets, and training logs.

---

## Setup Instructions

1. Clone the repo:

```bash
git clone https://github.com/yourusername/Defect-Detection.git
cd Defect-Detection
```
2. Create a virtual environment and install dependencies:


```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```
3. Run the FastAPI backend:


```
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
4. Run the Gradio UI:


```
python ui.py
```
5. Open the Gradio interface in your browser and upload steel/metal images for defect detection.




---

Screenshots


Place screenshots in a folder called screenshots/ and reference them here:


![1](https://raw.githubusercontent.com/JanithDoesCode/Steel-Metal-Manufacturing-Defect-Detection-System/blob/main/screenshots/1.png)
![2](https://raw.githubusercontent.com/JanithDoesCode/Steel-Metal-Manufacturing-Defect-Detection-System/blob/main/screenshots/2.png)
![3](https://raw.githubusercontent.com/JanithDoesCode/Steel-Metal-Manufacturing-Defect-Detection-System/blob/main/screenshots/3.png)

---

Notes

The system is built for Chamal Steel Fabricators.

This folder is fully reproducible, meaning anyone can run it with the model and dependencies provided.

Dataset is not included in this repo due to size constraints.

ML Workflow folder is for reference only — not required to run the system.
