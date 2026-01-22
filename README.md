# Diamond-Dynamics
💎 End-to-end Machine Learning project to predict diamond prices and segment diamonds into market categories using ML, ANN, K-Means clustering, PCA, and a Streamlit web app.
💎 Diamond Dynamics: Price Prediction & Market Segmentation
📌 Project Overview

The diamond market depends on multiple quality attributes such as carat, cut, clarity, and color to determine pricing.
This project builds an end-to-end Machine Learning system to:

🔹 Predict diamond prices in INR

🔹 Segment diamonds into meaningful market groups

🔹 Deploy predictions using a Streamlit web application

🎯 Objectives

Predict diamond prices using:

Linear Regression

Random Forest Regressor

Artificial Neural Network (ANN)

Segment diamonds into market clusters using:

K-Means Clustering

PCA for dimensionality reduction & visualization

Build a Streamlit UI for:

Price prediction

Market segment prediction

📊 Dataset

Source: Kaggle Diamonds Dataset

Shape: 53,940 rows × 10 features

Features
Column	Description
carat	Weight of the diamond (in carats)
cut	Quality of cut (Fair, Good, Very Good, Premium, Ideal)
color	Diamond color grading (D–J)
clarity	Measure of inclusions (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
depth	Total depth percentage
table	Width of top facet (%)
price	Price in USD
x	Length (mm)
y	Width (mm)
z	Depth (mm)
🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

TensorFlow / Keras

Streamlit

⚙️ Project Workflow

Data Loading & Cleaning

Exploratory Data Analysis (EDA)

Feature Engineering

Encoding of Categorical Variables

Regression Models (ML + ANN)

Model Evaluation (MAE, RMSE, R²)

K-Means Clustering

PCA Visualization

Cluster Naming

Model Saving (.pkl / .h5)

Streamlit Deployment

🧪 Feature Engineering

Derived features include:

Volume = x × y × z

Price per Carat = price / carat

Dimension Ratio = (x + y) / (2 × z)

Price Conversion: USD → INR

📈 Models Used
Regression

Linear Regression

Random Forest Regressor

Artificial Neural Network (ANN)

Clustering

K-Means

PCA for dimensionality reduction

📊 Model Evaluation

Metrics used:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

🧩 Market Segments

Clusters were labeled based on average carat and price:

Premium Heavy Diamonds

Mid-range Balanced Diamonds

Affordable Small Diamonds

🌐 Streamlit Web App

The app allows users to:

Input diamond attributes

Predict diamond price in INR

Predict market segment

Run the App
pip install streamlit pandas scikit-learn numpy
streamlit run app.py

📁 Repository Structure
diamond_app/
│
├── app.py
├── Diamond_Dynamics.ipynb
├── price_model.pkl
├── ann_price_model.h5
├── cluster_model.pkl
├── scaler.pkl
├── encoder.pkl
├── cluster_name_map.pkl
├── README.md

🚀 Future Enhancements

Hyperparameter tuning

Additional clustering algorithms

Cloud deployment

Enhanced Streamlit UI

👤 Author

Harsha Chourey
Aspiring Data Scientist | Machine Learning Enthusiast
