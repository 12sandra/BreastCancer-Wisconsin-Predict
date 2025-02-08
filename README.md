# BreastCancer-Wisconsin-Predict
Breast cancer prediction using machine learning from Wisconsin dataset
# Breast Cancer Prediction (Wisconsin Dataset)

## 📌 Project Overview
This project aims to predict breast cancer using the **Wisconsin Breast Cancer Dataset**. It utilizes **machine learning** techniques to classify tumors as **benign** or **malignant** based on various features extracted from cell nuclei.

## 📂 Dataset
The **Wisconsin Breast Cancer Dataset** is sourced from the UCI Machine Learning Repository. It contains **569 samples** with **30 numerical features** computed from digitized images of fine needle aspirate (FNA) biopsies of breast masses.

- **Target Variable**: Diagnosis (Benign - 0, Malignant - 1)
- **Features**: Mean, standard error, and worst (largest) values of ten real-valued features for each cell nucleus
- **Missing Values**: Some null values exist and need to be handled

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Algorithm**: Logistic Regression, Random Forest, Support Vector Machine (SVM), and Neural Networks

## 📊 Exploratory Data Analysis (EDA)
- Checked for missing values and handled them appropriately
- Visualized data distributions using histograms and box plots
- Used correlation heatmaps to identify feature importance

## 🚀 Model Training & Evaluation
1. **Data Preprocessing**:
   - Normalization and feature scaling
   - Train-test split (80%-20%)
2. **Model Training**:
   - Applied multiple machine learning models
   - Used cross-validation to avoid overfitting
3. **Performance Metrics**:
   - Accuracy, Precision, Recall, F1-score, ROC Curve

## 📈 Results
- Achieved high accuracy (>95%) with **Random Forest** and **SVM** models
- Feature importance analysis showed that certain cell nucleus features strongly influence classification

## 💡 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BreastCancer-Wisconsin-Predict.git
   cd BreastCancer-Wisconsin-Predict
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and run `breast_cancer_prediction.ipynb`

## 📜 Future Improvements
- Implement deep learning models (CNN, ANN) for improved performance
- Develop a web-based app using Flask/Streamlit for easy deployment
- Experiment with feature engineering techniques

## 🏷️ License
This project is open-source under the **MIT License**.

## 🤝 Contributing
Feel free to submit issues and pull requests to improve this project!

## 📬 Contact
For queries, reach out at **sandrasunilkumar4860@gmail.com** or open an issue on GitHub!

