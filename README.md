# 🌍 Supervised Learning for City-Level IP Geolocation  

### 🧠 Problem Statement 15 | AIORI-2 Remote Hackathon 2025  

---

### 👥 Team GEOSTHIRA  
**College:** Vemana Institute of Technology, Bengaluru  

**Team Members:**  
- 🧩 **Kavyashree K**  
- ⚙️ **Margaret Sheela C**  
- 🧑‍🏫 **Sneha Zolgikar** *(Internal Mentor)*  

---

## 🚧 Project Status  

> **Note:** This project is currently in progress.  
> The results and models shared here are **preliminary** — we are actively improving **model confidence, unseen IP prediction accuracy**, and **feature optimization**.  
> Future updates will include new datasets, retrained models, and confidence calibration reports.

---

## 📌 Project Overview  

The project **“Supervised Learning for City-Level IP Geolocation”** aims to predict the **city-level location** of a given IP address using **supervised machine learning techniques**.  
Our model learns patterns from IP octets, RTT statistics, DNS info, and network-related features to classify IPs by city.  

We built an **interactive Gradio dashboard** for live prediction — users can enter IPs (single, multiple, or via CSV) and view the predicted city and model confidence instantly.

---

## 🚀 Key Features  

✅ **Complete ML Pipeline** – from raw dataset → cleaned → synthesized → trained model  
✅ **Random Forest Classifier** for robust and interpretable predictions  
✅ **Feature Engineering** – RTT stats, octets, private/public flag, IP class  
✅ **Gradio Dashboard** – real-time predictions and confidence visualization  
✅ **Handles Private IPs** gracefully (“Private IP: Location cannot be determined”)  
✅ **Modular Codebase** – easy to retrain or extend with new features  

---

## 📦 Project Directory Structure

GEOSTHIRA_IP_GEOLOCATION/  
│  
├── 📁 Datasets/  
│   ├── AIORI_portal_data.csv  
│   ├── AIORI_portal_data_cleaned.csv  
│   ├── AIORI_portal_data_filtered.csv  
│   ├── AIORI_portal_data_synthesized.csv  
│   └── AIORI_portal_features_extracted.csv  
│  
├── 📁 src/  
│   ├── Data_Cleaning.ipynb  
│   ├── Data_Filtering.ipynb  
│   ├── Data_Synthesizing.ipynb  
│   ├── Feature_Extraction.ipynb  
│   ├── RandomForest_Training.ipynb  
│   └── Model_Prediction_and_Dashboard.ipynb  
│  
├── 📄 Geosthira_report_Oct2nd.pdf  
├── 📄 Geosthira_updated_report_Oct15th.pdf  
├── 📄 Proposed-structure-hackathon.pdf  
└── 📄 README.md  

---

## ⚙️ Workflow Summary  

| Stage | Notebook | Description |
|-------|-----------|-------------|
| 🧹 **Data Cleaning** | `Data_Cleaning.ipynb` | Removes nulls, duplicates, and invalid IPs |
| 🔍 **Filtering** | `Data_Filtering.ipynb` | Keeps only valid public IPs and relevant columns |
| 🧬 **Synthesizing** | `Data_Synthesizing.ipynb` | Balances city data and creates synthetic samples |
| 🧠 **Feature Extraction** | `Feature_Extraction.ipynb` | Extracts IP structure, RTT, and network features |
| 🌳 **Model Training** | `RandomForest_Training.ipynb` | Trains Random Forest model and saves `.pkl` |
| 🎛️ **Dashboard & Prediction** | `Model_Prediction_and_Dashboard.ipynb` | Gradio app for interactive predictions |

---

## 🧩 Model Details  

- **Algorithm:** Random Forest Classifier  
- **Framework:** scikit-learn  
- **Trained On:** Cleaned and balanced IP feature dataset  
- **Target:** City-level classification  
- **Confidence Metric:** Softmax probabilities via `predict_proba()`  

---

## 💻 Run the Project  

1. **Clone this repository**
   ```bash
   git clone https://github.com/geosthira-prog/AIORI-2-HACKATHON-PROJECTS.git
   cd AIORI-2-HACKATHON-PROJECTS

| Stage                          | Notebook                               | Description                                      |
| ------------------------------ | -------------------------------------- | ------------------------------------------------ |
| 🧹 **Data Cleaning**           | `Data_Cleaning.ipynb`                  | Removes nulls, duplicates, and invalid IPs       |
| 🔍 **Filtering**               | `Data_Filtering.ipynb`                 | Keeps only valid public IPs and relevant columns |
| 🧬 **Synthesizing**            | `Data_Synthesizing.ipynb`              | Balances city data and creates synthetic samples |
| 🧠 **Feature Extraction**      | `Feature_Extraction.ipynb`             | Extracts IP structure, RTT, and network features |
| 🌳 **Model Training**          | `RandomForest_Training.ipynb`          | Trains Random Forest model and saves `.pkl`      |
| 🎛️ **Dashboard & Prediction** | `Model_Prediction_and_Dashboard.ipynb` | Gradio app for interactive predictions           |


## 2.Open in Google Colab or Jupyter Notebook

## 3.Run notebooks in sequence:
1️⃣ Data_Cleaning.ipynb  
2️⃣ Data_Filtering.ipynb  
3️⃣ Data_Synthesizing.ipynb  
4️⃣ Feature_Extraction.ipynb  
5️⃣ RandomForest_Training.ipynb  
6️⃣ Model_Prediction_and_Dashboard.ipynb

## 4.Launch the Gradio dashboard
demo.launch()
Use Single IP, Multiple IPs, or CSV upload mode for predictions.

## 📊 Current Status
This is a **preliminary version** of our work.  
We are still improving:
- Model confidence and prediction stability  
- Error bounds for each city-level prediction  
- Handling of private IP ranges more robustly  

Future updates will include:
- Refined feature selection  
- Additional real-world IP datasets  
- Deployment-ready API version  

---

## 🧩 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Gradio  
- Matplotlib  

---

*With Regards*

Developed by *Team GEOSTHIRA*
under the guidance of Prof. Sneha Zolgikar
Vemana Institute of Technology, Bengaluru.

## 💬 “This is an ongoing research-driven project — results will evolve as we refine features and retrain our model.”


