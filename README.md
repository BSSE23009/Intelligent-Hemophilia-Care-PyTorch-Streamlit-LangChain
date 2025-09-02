# Hemophilia Care AI 🩸  

This project provides an AI-powered tool for assisting with **hemophilia care**.  
It includes:  

- 🧠 **Deep learning bruise detection model** (can be loaded with your own trained weights).  
- 📄 **Q&A module** to answer hemophilia-related questions using documents.  
- 🎛️ **Streamlit interface** for easy interaction.  

---

## 🚀 How to Run  

### 1. Install dependencies  
```bash
pip install -r requirements.txt
```

### 2. Add your model weights  
Place your trained PyTorch weights (e.g., `best_model.pth`) in the project root.  
> ⚠️ The file is not included in this repository for size/privacy reasons.  

### 3. Start the app  
```bash
streamlit run app.py
```

---

## 📂 Project Structure  
```
├── app.py                # Main Streamlit application  
├── requirements.txt      # Dependencies  
├── trials.ipynb          # Training & experiments notebook (optional)  
├── book/                 # Reference material (PDFs)  
└── bruises_detection/    # Dataset (train/test)  
```

---

## 📖 Features  
- **Bruise detection model**: Classifies images into bruise vs. normal skin (requires your trained `.pth` file).  
- **Hemophilia knowledge base**: Uses medical documents for Q&A.  
- **Interactive UI**: Upload images, ask questions, and get instant results.  

---

## 🔒 Notes on Deployment  
- This app is configured for **private deployment** (no shared API keys).  
- For public versions, users must input their own API keys.  

---

## 📜 License  
This project is licensed under the MIT License.  
