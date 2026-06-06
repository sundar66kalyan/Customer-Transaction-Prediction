# \# 📊 Customer Transaction Prediction

# 

# \## 🎯 Project Overview

# 

# This project predicts whether a customer will make a future transaction using Machine Learning and Deep Learning techniques.

# 

# The dataset contains 200 anonymized numerical features along with a binary target variable.

# 

# Target:

# \- 0 → Customer will not make a transaction

# \- 1 → Customer will make a transaction

# 

# The objective is to help banks identify potential customers and improve targeted marketing strategies. The problem is a binary classification task similar to the Santander Customer Transaction Prediction challenge. :contentReference\[oaicite:0]{index=0}

# 

# \---

# 

# \## 🚀 Business Objective

# 

# Banks can use this model to:

# 

# \- Identify customers likely to make future transactions

# \- Improve customer engagement

# \- Optimize marketing campaigns

# \- Increase conversion rates

# \- Support data-driven decision making

# 

# \---

# 

# \## 📂 Dataset Information

# 

# | Item | Value |

# |--------|--------|

# | Records | 200,000 |

# | Features | 200 |

# | Target Classes | 2 |

# | Missing Values | 0 |

# | Duplicate Records | 0 |

# 

# \### Class Distribution

# 

# | Class | Percentage |

# |---------|-----------|

# | 0 | 89.95% |

# | 1 | 10.05% |

# 

# The dataset is highly imbalanced.

# 

# \---

# 

# \## ⚙️ Project Workflow

# 

# \### 1. Data Understanding

# 

# \- Dataset inspection

# \- Data types analysis

# \- Statistical summary

# 

# \### 2. Data Cleaning

# 

# \- Missing value check

# \- Duplicate check

# \- ID column removal

# 

# \### 3. Exploratory Data Analysis

# 

# \- Class imbalance analysis

# \- Feature distribution analysis

# \- Correlation analysis

# 

# \### 4. Data Preprocessing

# 

# \- Train-Test Split (80:20)

# \- Feature Scaling using StandardScaler

# 

# \### 5. Model Building

# 

# Implemented Models:

# 

# \- Logistic Regression

# \- Random Forest

# \- Decision Tree

# \- K-Nearest Neighbors

# \- Gaussian Naive Bayes

# \- XGBoost

# \- LightGBM

# \- CatBoost

# \- Artificial Neural Network (ANN)

# 

# \### 6. Model Evaluation

# 

# Metrics Used:

# 

# \- Accuracy

# \- Precision

# \- Recall

# \- F1 Score

# \- ROC-AUC

# 

# \### 7. Hyperparameter Tuning

# 

# \- RandomizedSearchCV

# \- GridSearchCV

# 

# \### 8. Feature Importance Analysis

# 

# Performed using:

# 

# \- Logistic Regression

# \- Random Forest

# \- XGBoost

# \- LightGBM

# \- CatBoost

# 

# \---

# 

# \## 🏆 Model Comparison

# 

# | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

# |---------|---------|---------|---------|---------|---------|

# | Logistic Regression | 0.7925 | 0.2974 | 0.7816 | 0.4309 | 0.8657 |

# | Random Forest | 0.8995 | 0.0000 | 0.0000 | 0.0000 | 0.8097 |

# | Decision Tree | 0.7262 | 0.1751 | 0.4647 | 0.2543 | 0.6371 |

# | Gaussian NB | 0.9198 | 0.7008 | 0.3525 | 0.4690 | 0.8860 |

# | XGBoost | 0.9156 | 0.8306 | 0.2012 | 0.3240 | 0.8824 |

# | LightGBM | 0.9186 | 0.8096 | 0.2485 | 0.3803 | 0.8910 |

# | CatBoost | 0.9157 | 0.8438 | 0.1975 | 0.3201 | 0.8855 |

# | ANN | 0.9107 | 0.6243 | 0.2799 | 0.3865 | 0.8505 |

# 

# \---

# 

# \## 🥇 Best Model

# 

# \### LightGBM

# 

# Performance:

# 

# \- Accuracy: 91.86%

# \- Precision: 80.96%

# \- Recall: 24.85%

# \- F1 Score: 38.03%

# \- ROC-AUC: 89.10%

# 

# Reasons for Selection:

# 

# ✅ Highest ROC-AUC

# 

# ✅ Fast Training

# 

# ✅ Handles High-Dimensional Data

# 

# ✅ Production Ready

# 

# \---

# 

# \## 🔍 Top Features

# 

# LightGBM Important Features:

# 

# 1\. var\_76

# 2\. var\_174

# 3\. var\_146

# 4\. var\_34

# 5\. var\_6

# 

# \---

# 

# \## 📈 Results

# 

# The LightGBM model achieved the highest ROC-AUC score and was selected as the final production model.

# 

# \---

# 

# \## 🛠 Technologies Used

# 

# \- Python

# \- Pandas

# \- NumPy

# \- Matplotlib

# \- Seaborn

# \- Scikit-Learn

# \- XGBoost

# \- LightGBM

# \- CatBoost

# \- TensorFlow / Keras

# 

# \---

# 

# \## 📁 Project Structure

# 

# Customer-Transaction-Prediction/

# 

# │

# 

# ├── Customer\_Transaction\_Prediction.ipynb

# 

# ├── customer\_transaction\_lightgbm.pkl

# 

# ├── Customer\_Transaction\_Report.pdf

# 

# ├── README.md

# 

# ├── requirements.txt

# 

# ├── feature\_importance.png

# 

# ├── roc\_curve\_comparison.png

# 

# ├── model\_comparison.png

# 

# └── images/

# 

# \---

# 

# \## 💾 Model Saving

# 

# ```python

# import joblib

# 

# joblib.dump(

# &#x20;   lgb,

# &#x20;   "customer\_transaction\_lightgbm.pkl"

# )

# ```

# 

# \---

# 

# \## 📌 Future Improvements

# 

# \- SHAP Explainability

# \- Threshold Optimization

# \- Ensemble Learning

# \- Streamlit Deployment

# \- MLOps Pipeline

# 

# \---

# 

# \## 👨‍💻 Author

# 

# Kalyanasundar 

# 

# GitHub:

# 

# https://github.com/sundar66kalyan

