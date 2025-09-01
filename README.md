# Titanic Survival AI 🚢🛟

An interactive web app built with **Streamlit** that predicts whether a passenger would have survived the Titanic disaster, based on their details.  
It uses **machine learning models** (like Logistic Regression, Random Forest, etc.) trained on the classic Titanic dataset.  

---

## 🔹 Features
- Upload a **training CSV** (with the `Survived` column) to train the model.
- Upload a **test CSV** (without `Survived`) or enter details of a single passenger.
- Get predictions with survival probability.
- View **Random Forest feature importance**.

---

## 🔹 Installation
Clone this repo:
```bash
git clone https://github.com/tlpranathi/titanic-survival-ai.git
cd titanic-survival-ai
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the App
```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually http://localhost:8501).