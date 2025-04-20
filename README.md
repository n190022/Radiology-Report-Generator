# Radiology-Report-Generator

#  Radiology Report Generator

A **Streamlit** web application that:

- **Classifies** medical images (X‑ray, MRI, CT) into one of five body‑part categories.
- **Predicts** specific diseases for each identified body part using a dedicated PyTorch model.
- **Generates** a formatted, radiology‑style PDF report with findings and recommendations.
- **Applies** custom CSS for a clean, professional UI.
- **Allows** users to download the report as a PDF.

---

## 🗄️ Project Structure

```
radiology_app/
├── app.py                   # Main Streamlit application with CSS styling
├── models/
│   ├── xrayctscan                 
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

## ⚙️ Prerequisites

- Python 3.8 or higher
- `pip` for package management

---
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure** the `models/` folder contains the correct `.pth` files with names matching those in `app.py`.

---

## 🚀 Running the App

From the project root, run:
```bash
streamlit run app.py
```

- **Upload** an X‑ray/MRI/CT image using the file uploader.
- The app **classifies** body part and disease.
- A **radiology report** displays on screen.
- Click **Download Radiology Report (PDF)** to save.

---

## 🎨 Custom CSS

The UI styling is injected directly via a `<style>` block in `app.py`. You can modify colors, fonts, and components by editing this section near the top of the file:

```python
st.markdown("""
<style>
  /* Add or adjust your CSS here */
</style>
""", unsafe_allow_html=True)
```

---

## 🔧 Configuration

- **Model paths**: Edit the `body_part_model_path` and `disease_model_paths` dictionaries in `app.py` to point to your trained model files.
- **Labels**: Update `disease_labels` in `app.py` to match the classes your models were trained on.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push: `git push origin feature/my-feature`.
5. Open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
