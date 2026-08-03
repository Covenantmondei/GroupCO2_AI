# Tomato Leaf Disease Classifier — Group C02

An artificial intelligence web application developed for **GET 324 (Computer Engineering Laboratory Course)** to diagnose **Tomato Mosaic Virus** from leaf images.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](http://groupco2ai.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Covenantmondei/GroupCO2_AI)

---

## Group C02 Team Members

| Name | Registration No. | Role | GitHub | Email |
| :--- | :---: | :--- | :---: | :--- |
| **Covenant Idorenyin Monday** | `23/EG/CO/030` | Lead ML Engineer (Transfer Learning & Cloud Deployment) | [@Covenantmondei](https://github.com/Covenantmondei) | covenantmonday863@gmail.com |
| **Manuel Ibup Efonema** | `23/EG/CO/130` | Project Coordinator, Git Version Control & Technical Writer | [@manuelibup](https://github.com/manuelibup) | manuelibup@gmail.com |
| **Etim Victory Paul** | `23/EG/CO/040` | Frontend Developer (Streamlit Web App & CSS Theme Design) | [@Vinel-Ux](https://github.com/Vinel-Ux) | - |
| **Libang Success Obi** | `23/EG/CO/010` | Data Pipeline Engineer (Splits, Caching, Shuffle & Prefetch) | [@libangsuccess-hub](https://github.com/libangsuccess-hub) | - |
| **Ekeruke Aniekan Ukpono** | `23/EG/CO/110` | CNN Architect (Custom 3-Block CNN Design & Baseline Training) | [@greataniekan](https://github.com/greataniekan) | - |
| **Okon Imoh Patrick** | `23/EG/CO/060` | QA & Model Evaluation Specialist (Metrics & Confusion Matrices) | - | imohpatrick17@gmail.com |
| **Bassey Nora Bassey** | `23/EG/CO/070` | EDA & Data Augmentation Analyst (Preprocessing & Visualizations) | [@basseynora](https://github.com/basseynora) | - |
| **Effiong Abasiodiong Ita** | `23/EG/CO/090` | System Integration & Local Testing Support | - | effiongabasiodiong3@gmail.com |
| **Abasi-ifreke Ukpong Obot** | `23/EG/CO/020` | Model Deployment Assistant & UI Testing | [@Abasifreke12](https://github.com/Abasifreke12) | - |
| **Ediomo Godwin Udontah** | `23/EG/CO/050` | Model Hyperparameter Tuning & Validation Support | [@01Ediomo](https://github.com/01Ediomo) | - |

---

## Project Overview

This project implements a complete end-to-end Machine Learning pipeline:
1. **Dataset Preparation**: Downloading, filtering, and splitting raw plant disease images.
2. **Data Pipeline Optimization**: Implementing image augmentation, caching, and prefetching.
3. **Model Development**: Building a custom CNN from scratch and developing a Transfer Learning model using pre-trained **MobileNetV3Small**.
4. **Evaluation**: Computing classification metrics, confusion matrices, and plotting learning curves.
5. **Deployment**: Building an interactive Streamlit dashboard and hosting it on Streamlit Community Cloud.

---

## Dataset Specifications

- **Source**: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Target Classes**: 
  - `Tomato_healthy` (Index 1 / Class: Healthy)
  - `Tomato__Tomato_mosaic_virus` (Index 0 / Class: Tomato Mosaic Virus)
- **Split Ratio**: 70% Train, 15% Validation, 15% Test (created using `split-folders`).
- **Data Pipeline Optimization**: Configured with `tf.data` API using `.cache()`, `.shuffle(1000)`, and `.prefetch(buffer_size=tf.data.AUTOTUNE)` to prevent hardware bottlenecking.

---

## Model Performance Summary

The deep learning models were trained for up to **50 epochs** with dynamic learning rate reduction (`ReduceLROnPlateau`) and early stopping (`EarlyStopping` with patience=8).

### Test Evaluation Metrics (Unseen Data)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 (Transfer Learning)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | 1,061,602 |
| **MobileNetV3 (Fine-Tuned)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | 1,061,602 (Top 30 layers unfrozen) |
| **Custom CNN (from scratch)** | *91.24%* | *0.8950* | *0.9020* | *0.8980* | 345,825 |

> [!NOTE]
> The MobileNetV3 Transfer Learning model achieved a validation accuracy of **99.66%** and validation loss of **0.0126** during feature extraction, which generalized perfectly to a **100% test accuracy** on the 297 unseen test images (57 Mosaic Virus, 240 Healthy).

---

## Local Setup & Installation

To run the Streamlit app locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Covenantmondei/GroupCO2_AI.git
cd GroupCO2_AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## Repository Structure
```
GroupCO2_AI/
├── models/
│   └── mobilenetv3_final.keras  # Saved fine-tuned Keras model
├── app.py                      # Streamlit application UI & prediction logic
├── readme.MD                   # Repository documentation
├── requirements.txt            # Python dependencies
└── tomato_ai.ipynb             # Jupyter Notebook with the model training pipeline
```
