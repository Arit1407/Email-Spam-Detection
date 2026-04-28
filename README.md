# 📩 Email Spam Detection API with Explainability

## 🚀 Overview

This project is an end-to-end Natural Language Processing (NLP) system that classifies emails as **Spam** or **Ham (Not Spam)** using Machine Learning.

The application accepts raw email text, preprocesses it using an NLP pipeline, converts it into numerical features, predicts whether the email is spam, and provides **human-understandable explanations** for every prediction.

The system is deployed using **FastAPI** and fully containerized using **Docker**.

---

## 🎯 Problem Statement

Spam emails are unwanted emails commonly used for:

- Promotions  
- Fraud attempts  
- Phishing attacks  
- Malware links  
- Fake offers  

Manual filtering is inefficient and time-consuming. This project automates spam email detection using NLP and Machine Learning.

The objective is to build a production-ready solution that:

- Detects spam emails accurately  
- Exposes predictions through an API  
- Provides explainable predictions  
- Runs inside Docker containers  

---

## 🧠 Features

- Email Spam / Ham Classification  
- Complete NLP Text Preprocessing Pipeline  
- TF-IDF Vectorization  
- Logistic Regression Classifier  
- Explainable AI Predictions  
- FastAPI REST API  
- Swagger Interactive Documentation  
- Dockerized Deployment  
- Modular and Scalable Code Structure  

---

## 🛠️ Tech Stack

- Python 3.10  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- BeautifulSoup  
- FastAPI  
- Uvicorn  
- Joblib  
- Docker  

---

## 🏗️ Project Structure

```text
email-spam-detector/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── data/
│   └── emails.csv
│
├── models/
│   └── spam_model.pkl
│
├── reports/
│   └── classification_report.txt
│
└── src/
    ├── data_ingestion.py
    ├── preprocessing.py
    ├── train.py
    └── explain.py