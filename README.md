# 🚗 CO2 Emission AI Prediction

![CO2 Emission Dashboard](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Welcome to the **CO2 Emission AI Prediction** repository! This project uses machine learning to classify a vehicle's CO2 emission levels into three categories: **Low, Medium, and High**. It includes a full end-to-end workflow, from data preprocessing and model training in a Jupyter Notebook to a beautifully designed, interactive web application built with Streamlit.

---

## 🌟 Features

- **Interactive Web Dashboard**: A user-friendly Streamlit interface where users can manually configure a vehicle's specifications or instantly generate random vehicle data to test the model.
- **AI Classification**: Predicts whether a vehicle's emissions will be `Low 🌱`, `Medium ⚖️`, or `High 🚨` based on engine specifications and fuel consumption.
- **Smart Recommendations**: Provides actionable eco-driving tips and vehicle suggestions based on the predicted emission class.
- **Visual Insights**: Live bar charts displaying the configured vehicle's fuel consumption metrics (City, Highway, Combined).

---

## 🛠️ Tech Stack & Requirements

- **Python** (Pandas, Numpy)
- **Scikit-Learn** (Machine Learning Pipeline, Scalers, Encoders)
- **Streamlit** (Web Application Framework)
- **Joblib** (Model Serialization)

---

## 📂 Repository Structure

```text
📁 CO2 Emssion/
├── 📄 app.py                                  # The main Streamlit web application script
├── 📄 requirements.txt                        # Python dependencies for deployment
├── 📓 co2 emssion (classfication).ipynb       # Jupyter Notebook with data exploration and model training
├── 📦 best_co2_emission_model.pkl             # Trained Scikit-Learn classification model
├── 📦 scalerf.pkl                             # Trained StandardScaler for feature normalization
├── 📦 fuel_encoder.pkl                        # Encoder for Fuel Type
├── 📦 transmission_encoder.pkl                # Encoder for Transmission
├── 📦 vehicle_class_encoder.pkl               # Encoder for Vehicle Class
├── 🖼️ confusion_matrix.png                    # Evaluation metric visualization
└── 🖼️ model_comparison.png                    # Model comparison visualization
```

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/co2-emssion.git
   cd co2-emssion
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open in your default web browser.*

---

## 🧠 How the Model Works

The model analyzes key vehicle metrics to estimate its environmental impact. The input features include:
- **Vehicle Class** (e.g., SUV, Compact, Mid-size)
- **Engine Size** (in Liters)
- **Cylinders** (3, 4, 6, 8, 12)
- **Transmission Type** (Automatic, Manual, etc.)
- **Fuel Type** (Regular, Premium, Diesel, Ethanol)
- **Fuel Consumption** (City, Highway, and Combined measured in L/100km)

These features are encoded and scaled using the pre-trained `joblib` objects before being passed to the Random Forest / Decision model (`best_co2_emission_model.pkl`) for inference.

---

## 📊 Model Evaluation

During training, several models were tested. The final model was selected based on optimal accuracy and generalization. 

*(Check the `confusion_matrix.png` and `model_comparison.png` files included in this repo for a visual breakdown of the training results!)*

---

### 🤝 Contributing
Feel free to open an issue or submit a pull request if you have ideas on how to improve the predictions or the UI!
