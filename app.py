# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fpdf import FPDF
import os

# Paths
body_part_model_path = "C:\\Users\\sangeetha\\class_2"
disease_model_paths = {
    "Bone_Fracture_Binary_Classification": "C:\\Users\\sangeetha\\bone_10",
    "brain-tumor": "C:\\Users\\sangeetha\\model_16",
    "bone": "C:\\Users\\sangeetha\\boneknee_20",
    "alzheimer_mri": "C:\\Users\\sangeetha\\modelalzhe_2",
    "chest": "C:\\Users\\sangeetha\\modelchest_19"
}
disease_labels = {
    "Bone_Fracture_Binary_Classification": ["fractured", "not fractured"],
    "brain-tumor": ["Glioma", "Pituitary", "Meningioma", "No Tumor", "Other"],
    "bone": ["Osteoporosis", "Normal", "Osteopenia"],
    "alzheimer_mri": ["Non Demented", "Mild Dementia", "Very mild Dementia", "Moderate Dementia"],
    "chest": ["TUBERCULOSIS", "NORMAL", "PNEUMONIA"]
}

# Model class
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 6 * 128, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
@st.cache_resource
def load_model(model_path, num_classes):
    model = MyModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict body part
def predict_body_part(image_path):
    label_dict = {
        0: "Bone_Fracture_Binary_Classification",
        1: "brain-tumor",
        2: "bone",
        3: "alzheimer_mri",
        4: "chest"
    }
    model = load_model(body_part_model_path, len(label_dict))
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return label_dict[pred.item()]

# Predict disease
def predict_disease(image_path, body_part):
    model_path = disease_model_paths.get(body_part)
    labels = disease_labels.get(body_part)
    if not model_path or not labels:
        return "Unknown"
    model = load_model(model_path, len(labels))
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return labels[pred.item()]

# PDF Report
def generate_pdf(body_part, disease):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Radiology Report", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Body Part Predicted: {body_part}")
    pdf.multi_cell(0, 10, f"Predicted Disease/Condition: {disease}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Recommendation: Please consult a radiologist or physician for professional evaluation.")
    pdf.output("radiology_report.pdf")
    return "radiology_report.pdf"

# Streamlit UI
st.set_page_config(page_title="Radiology Predictor", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stFileUploader {
        padding: 1rem;
        border: 2px dashed #1f77b4;
        border-radius: 12px;
        background-color: #ffffff;
        margin-top: 20px;
    }
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    .stDownloadButton button:hover {
        background-color: #45a049;
    }
    img {
        border: 1px solid #ccc;
        border-radius: 12px;
        padding: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 AI Radiology Report Generator")

uploaded_file = st.file_uploader("Upload an X-ray/MRI/CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    body_part = predict_body_part("temp.jpg")
    disease = predict_disease("temp.jpg", body_part)

    st.success(f"🧠 Predicted Body Part: {body_part}")
    st.success(f"🦠 Predicted Condition: {disease}")

    pdf_file = generate_pdf(body_part, disease)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Radiology Report (PDF)",
            data=f,
            file_name=pdf_file,
            mime="C:\\Users\\sangeetha"
        )
