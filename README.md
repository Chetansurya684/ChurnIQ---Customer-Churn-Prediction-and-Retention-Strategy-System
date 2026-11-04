# ChurnIQ – Customer Churn Prediction and Retention Strategy System

**An Intelligent Machine Learning-Based Customer Churn Prediction and Retention Platform**

---

## 📌 Overview

ChurnIQ is a machine learning-based customer analytics system designed to predict customer churn and support effective customer retention strategies, particularly in the telecommunications industry.

The system analyzes customer information, service subscriptions, billing details, contract types, payment methods, and behavioral attributes to identify customers who are likely to discontinue a service.

ChurnIQ integrates data preprocessing, exploratory data analysis, machine learning model comparison, churn probability prediction, customer risk assessment, and an interactive dashboard into a unified platform.

It also provides rule-based retention recommendations that help business managers and customer relationship executives identify high-risk customers and plan targeted retention activities.

## 🎯 Objectives

- Develop a machine learning system to predict customer churn.
- Analyze customer data to identify patterns associated with customer attrition.
- Evaluate multiple machine learning classification algorithms.
- Identify customers based on their predicted churn probability.
- Classify customers into different risk categories.
- Provide actionable retention recommendations.
- Visualize customer churn insights through an interactive dashboard.
- Support data-driven customer relationship management.

## ✨ Key Features

### 1. Customer Churn Prediction
- Predicts whether a customer is likely to leave a service.
- Calculates estimated churn probability.
- Supports individual customer risk analysis.

### 2. Customer Risk Assessment
- Identifies customers with potential churn risk.
- Categorizes customers into High, Medium, and Low Risk.
- Provides customer-level risk information.
- Supports filtering and searching customer records.

### 3. Machine Learning Model Comparison
- Implements multiple classification algorithms.
- Compares model performance using standard evaluation metrics.
- Helps identify a suitable model based on experimental results.

### 4. Retention Strategy Engine
- Provides rule-based retention recommendations.
- Suggests personalized offers and loyalty benefits.
- Helps identify customers requiring immediate attention.
- Supports customer engagement and retention planning.

### 5. Interactive Dashboard
- Displays customer statistics and churn distribution.
- Visualizes customer risk categories.
- Presents churn trends based on contract types.
- Provides model performance comparisons.
- Helps business users interpret analytical results.

### 6. MySQL Data Management
- Supports MySQL database connectivity.
- Provides database initialization options.
- Supports customer data management.
- Includes prediction history functionality.

### 7. Project Information
- Provides project objectives and problem statement.
- Displays the technology stack.
- Explains the system workflow and major functionalities.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Machine Learning Models | Logistic Regression, Random Forest, HistGradientBoosting |
| Database | MySQL |
| Data Visualization | Streamlit Charts and Visualization |

## 🧠 Machine Learning Models

ChurnIQ evaluates different supervised machine learning classification algorithms to identify customers who are likely to churn.

The research paper presents the following models:

1. **Logistic Regression:** A classification algorithm used to estimate the probability of customer churn.
2. **Random Forest:** An ensemble learning algorithm that combines multiple decision trees to make predictions.
3. **HistGradientBoosting:** A gradient boosting-based classification algorithm used to identify patterns in customer data.

### Model Evaluation Metrics

The models are evaluated using the following performance metrics:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many predicted churn cases are actually churn cases.
- **Recall:** Measures how many actual churn cases are correctly identified.
- **F1-Score:** Represents the balance between precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between churn and non-churn customers.

## 📊 Experimental Results

The following results are reported in the research paper's model comparison table.

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6600 | 0.7453 | 0.6630 | 0.7018 | 0.7276 |
| Random Forest | 0.6500 | 0.7346 | 0.6575 | 0.6939 | 0.6576 |
| HistGradientBoosting | 0.6233 | 0.6795 | 0.7127 | 0.6954 | 0.6150 |

### Best Performing Model

Based on the reported ROC-AUC results, **Logistic Regression** achieved the highest score of 0.7276 and was selected as the preferred model in the research paper.

The results demonstrate the importance of comparing different classification algorithms rather than selecting a model based only on accuracy.

## 🔄 Proposed System Methodology

ChurnIQ follows a systematic machine learning workflow:

```text
Customer Data Collection
          |
          v
    Data Preprocessing
          |
          v
Exploratory Data Analysis
          |
          v
    Feature Selection
          |
          v
  Model Development
          |
          v
    Model Evaluation
          |
          v
  Churn Prediction
          |
          v
 Customer Risk Assessment
          |
          v
 Retention Recommendations
          |
          v
 Interactive Dashboard
          |
          v
 Business Decision Making
```

### Workflow Explanation

1. **Customer Data Collection:** Collects customer information, including tenure, contract type, billing details, service subscriptions, and payment methods.

2. **Data Preprocessing:** Handles missing values, duplicate records, inconsistent data, and categorical attributes.

3. **Exploratory Data Analysis:** Examines customer characteristics, churn distribution, and relevant patterns.

4. **Feature Selection:** Identifies important customer attributes for machine learning model development.

5. **Model Development:** Trains selected classification algorithms using historical customer data.

6. **Model Evaluation:** Evaluates model performance using accuracy, precision, recall, F1-score, and ROC-AUC.

7. **Churn Prediction:** Estimates the probability that a customer may discontinue a service.

8. **Risk Assessment:** Organizes customers into risk categories based on their predicted churn probability.

9. **Retention Recommendations:** Provides rule-based suggestions to support customer retention planning.

10. **Dashboard Visualization:** Presents prediction results, customer insights, and analytical information through an interactive interface.

## 🧩 Major System Modules

| Module | Description |
|---|---|
| Dashboard | Displays customer statistics, churn rate, and risk distribution. |
| Predict Churn | Predicts individual customer churn probability. |
| Customer Risk Center | Identifies and filters customers based on risk levels. |
| Model Performance | Compares machine learning model evaluation metrics. |
| Retention Strategies | Provides risk-based retention recommendations. |
| MySQL / Data Management | Supports database connectivity and data management. |
| Project Info | Presents project objectives, methodology, and technical information. |

## 📈 Dashboard Highlights

The ChurnIQ dashboard provides an interactive interface for analyzing customer churn and retention opportunities.

The research paper presents the following dashboard statistics:

| Dashboard Metric | Reported Value |
|---|---:|
| Total Customers | 1,500 |
| Actual Churned Customers | 907 |
| Overall Churn Rate | 60.5% |
| Customers Identified as At Risk | 576 |
| Average Monthly Charge | ₹85 |

### Dashboard Functionalities

- Customer churn overview and statistics.
- Individual customer churn probability analysis.
- Customer risk identification and filtering.
- Machine learning model performance comparison.
- Risk-based retention strategy recommendations.
- MySQL connectivity and data management.

## 💼 Business Applications

ChurnIQ can support organizations in:

- Telecommunications customer relationship management.
- Customer retention and engagement planning.
- Identification of high-risk customer segments.
- Customer behavior analysis.
- Data-driven business decision-making.
- Targeted customer retention campaigns.

## 🔮 Future Scope

The system can be extended with the following enhancements:

- Real-time customer data integration.
- Advanced machine learning and deep learning models.
- Personalized retention recommendation systems.
- Automated customer notification and engagement.
- Integration with CRM platforms.
- Regular model retraining and performance monitoring.
- Advanced analytics for customer behavior and retention trends.

## ⚠️ Limitations

- Prediction performance depends on the quality and characteristics of the available dataset.
- Model performance may change when applied to different customer populations.
- Retention recommendations are decision-support suggestions and do not guarantee customer retention.
- Real-time data integration and automated customer communication are considered future enhancements.
- Further validation using external datasets and real-world retention campaigns is required.

## 👨‍💻 Research & Development Team

**Project Title:** ChurnIQ – Customer Churn Prediction and Retention Strategy System

**Institution:** R. C. Patel Institute of Technology, Shirpur, Maharashtra, India

**Department:** Computer Engineering

**Authors:**

- **Punam R. Patil** – Assistant Professor, Department of Computer Engineering
- **Chetan Lahu Suryawanshi** – B.Tech Student, Department of Computer Engineering
- **Lokesh Vikasrao Bhadane** – B.Tech Student, Department of Computer Engineering

## 📚 Research References

1. S. Ouf, K. T. Mahmoud, and M. A. Abdel-Fattah, "A Proposed Hybrid Framework to Improve the Accuracy of Customer Churn Prediction in Telecom Industry," *Journal of Big Data*, vol. 11, Art. no. 70, 2024.

2. A. Barsotti et al., "A Decade of Churn Prediction Techniques in the TelCo Domain: A Survey," *SN Computer Science*, vol. 5, Art. no. 404, 2024.

3. V. Chang et al., "Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models," *Algorithms*, vol. 17, no. 6, Art. no. 231, 2024.

4. A. Sikri et al., "Enhancing Customer Retention in Telecom Industry with Machine Learning Driven Churn Prediction," *Scientific Reports*, vol. 14, Art. no. 13097, 2024.

5. L. Saha et al., "Deep Churn Prediction Method for Telecommunication Industry," *Sustainability*, vol. 15, no. 5, Art. no. 4543, 2023.

6. M. Z. Alotaibi and M. A. Haq, "Customer Churn Prediction for Telecommunication Companies Using Machine Learning and Ensemble Methods," *Engineering, Technology & Applied Science Research*, vol. 14, no. 3, pp. 14572–14578, 2024.

7. P. Wanikar et al., "Telco Customer Churn Prediction Using ML Models," *International Journal of Intelligent Systems and Applications in Engineering*, vol. 12, no. 2, 2024.

8. A. Khattak et al., "Customer Churn Prediction Using Composite Deep Learning Technique," *Scientific Reports*, vol. 13, Art. no. 17294, 2023.

---

## 📄 License

This project is intended for academic and research purposes. Add an appropriate open-source license if you intend to distribute or reuse the source code publicly.

---

**ChurnIQ – Turning Customer Data into Actionable Retention Insights.**
