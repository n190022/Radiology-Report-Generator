# Radiology-Report-Generator

#  Radiology Report Generator


This application leverages deep learning models to generate diagnostic reports based on uploaded medical images (X-rays or CT scans). It classifies the images into different categories, predicts the body part or scan type, and generates a disease diagnosis. The app provides a downloadable diagnostic report that includes the predicted results.

A **Streamlit** web application that:

- **Classifies** medical images (X‑ray, MRI, CT) into one of five body‑part categories.
- **Predicts** specific diseases for each identified body part using a dedicated PyTorch model.
- **Generates** a formatted, radiology‑style PDF report with findings and recommendations.
- **Applies** custom CSS for a clean, professional UI.
- **Allows** users to download the report as a PDF.

---

Features
Image Classification: The model identifies whether the uploaded image is an X-ray or CT scan.

Body Part Prediction: Once the image is identified as an X-ray or CT scan, the app predicts the body part or scan type (e.g., bone fracture, brain tumor, etc.).

Disease Prediction: The app predicts the disease associated with the body part.

Diagnostic Report Generation: The app generates a report with the prediction results, which can be downloaded as a .txt file.


Requirements
Python 3.x

PyTorch: For model inference.

Streamlit: For the web interface.

Pillow: For image processing.

torchvision: For image transformations.




## 🗄️ Project Structure

```
radiology_app/
├── app.py                   # Main Streamlit application with CSS styling
├── models/
│   ├── xrayctscan       #checks whether image is xrayctscan or not           
│   ├── class_2          # Body‑part classifier
│   ├── bone_10          # Bone fracture classifier
│   ├── model_16         # Brain tumor classifier
│   ├── boneknee_20      # Bone (knee) classifier
│   ├── modelalzhe_2     # Alzheimer MRI classifier
│   └── modelchest_19    # Chest disease classifier
├── requirements.txt         # Python dependencies
└── README.md                # This documentation file
```

---

How to Use
Upload Image: Click on the "Upload an image" button to select an image file (supported formats: .jpg, .jpeg, .png).

View Results: The app will display the uploaded image and predict whether it is an X-ray or CT scan.

Body Part and Disease Prediction: After identifying the scan type, the app predicts the body part (e.g., bone fracture, brain tumor) and the disease associated with it.

Download Report: After predictions are made, the app generates a diagnostic report. You can download the report as a .txt file by clicking the "Download Report" button.

File Structure
PROJECT.py: The main Streamlit application file containing the image processing, model inference, and report generation logic.



## 🚀 Running the App

From the project root, run:
```bash
streamlit run PROJECT.py
```

- **Upload** an X‑ray/MRI/CT image using the file uploader.
- The app **classifies** body part and disease.
- A **radiology report** displays on screen.
- Click **Download Radiology Report (PDF)** to save.

---


## 🔧 Configuration

- **Model paths**: Edit the `body_part_model_path` and `disease_model_paths` dictionaries in `PROJECT.py` to point to your trained model files.
- **Labels**: Update `disease_labels` in `PROJECT.py` to match the classes your models were trained on.

---


REQUIREMENTS
streamlit
torch
torchvision
pillow
numpy





HOW TO DO :

Here's a step-by-step guide to help you set up the environment and run the app without using a virtual environment.

Step 1: Install Python
Make sure that Python 3.x is installed on your system. You can download it from Python's official website.

After installing Python, verify the installation by running the following command in your terminal or command prompt:

Step 2: Install Required Libraries
Since you're not using a virtual environment, the libraries will be installed globally on your system. Use pip to install the necessary libraries directly:

Install Streamlit, PyTorch, and other required libraries:

STEP 3 :
Prepare the Models
You need pre-trained models for scan type classification, body part classification, and disease prediction. Ensure these models are saved locally and are accessible.
Step 3: Prepare the Models
Ensure the models are saved in the correct location:

Main scan classification model (xrayctscan vs others).

Body part classification model.

Disease-specific models for body parts (e.g., chest, brain, etc).

Update Model Paths: Ensure the model paths in your code match the locations where your models are saved. For example:

disease_model_paths = {
    "Bone_Fracture_Binary_Classification": "C:\\Users\\sangeetha\\bone_10",
    "brain-tumor": "C:\\Users\\sangeetha\\model_16",
    "bone": "C:\\Users\\sangeetha\\boneknee_20",
    "alzheimer_mri": "C:\\Users\\sangeetha\\modelalzhe_4",
    "chest": "C:\\Users\\sangeetha\\modelchest_19",
}
Ensure the models are properly trained and saved in appropriate format.

Step 4: Run the Streamlit App
Create the app.py file: Save the code you’ve provided (from the original code you shared) into a file named app.py.

Run the Streamlit App: Navigate to the directory where the app.py file is located and run:

streamlit run PROJECT.py


Step 5: Upload and Process an Image
Upload an Image:

Go to the Streamlit app page in your browser (http://localhost:8501).

Use the file uploader to upload an X-ray or CT scan image in .jpg, .jpeg, or .png format.

Processing:

The app will check whether the image is an X-ray or CT scan.

It will predict the body part (e.g., chest, brain, bone).

It will then predict the disease based on the body part.

A diagnostic report will be generated and displayed, including options to download it.

Step 6: Diagnostic Report Generation
Once the image is processed:

The app will display the predicted body part and predicted disease.

It will also generate a diagnostic report in markdown format, including an option to download the report as a .txt file.

