# Heart Disease Prediction Project

This project aims to predict heart disease using various machine learning models. The models used in this project include Logistic Regression, Naive Bayes, Random Forest Classifier, Extreme Gradient Boost, K-Nearest Neighbors, XGBoost, and Support Vector Classifier (SVC). The project evaluates these models based on their accuracy and other performance metrics.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)


## Installation

To run this project, you'll need to have Python installed along with several libraries. You can install the necessary libraries using the following command:

pip install numpy pandas scikit-learn matplotlib seaborn xgboost
## Dataset
The dataset used for this project is the Heart Disease dataset. Ensure you have the dataset in a CSV format. You can download the dataset from this Repository

## Features
- age: Age of patient
- sex: Male or female
- chest pain type: 4 values
- resting blood pressure: Resting blood pressure of patient
- serum cholestoral: Serum cholesterol in mg/dl
- fasting blood sugar: > 120 mg/dl
- resting electrocardiographic results: Values 0, 1, 2
- maximum heart rate achieved
- exercise induced angina
- oldpeak: ST depression induced by exercise relative to rest
- slope: The slope of the peak exercise ST segment
- number of major vessels: (0-3) colored by fluoroscopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversible defect
- There are no null values in the dataset.

## Exploratory Data Analysis (EDA)
There is no strong correlation between two features in the given dataset.
Age is normally distributed.
In the given dataset, half of the people have heart disease and half do not.
Men have more heart disease compared to women.
## Usage
Clone the repository:
bash
Copy code
git clone https://github.com/Jyant1008/heart-disease-prediction.git
Navigate to the project directory:
bash
Copy code
cd heart-disease-prediction
Open the Jupyter Notebook:
bash
Copy code
jupyter notebook " jupyter notebook heart disease prediction.ipynb"
Run all the cells in the notebook to train and evaluate the models.
## Models Used
- Logistic Regression
- Naive Bayes
- Random Forest
- Extreme Gradient Boost
- K-Nearest Neighbors
- XGBoost
- Support Vector Classifier (SVC)
- Best Model
- The XGBoost classifier model provided the best result with 97% accuracy.
- Hyperparameter tuning was applied to the XGBoost model to further improve its performance.
## Evaluation Metrics
The models are evaluated based on the following metrics:

- Accuracy Score: The ratio of correctly predicted instances to the total instances.
- Confusion Matrix: A table to describe the performance of a classification model.
- ROC Curve: A graphical plot illustrating the diagnostic ability of a binary classifier.
- Classification Report: A detailed report showing precision, recall, F1-score, and support.
## Results
The results section of the notebook contains visualizations and performance metrics for the models used. The XGBoost model provided the highest accuracy at 97%. Hyperparameter tuning slightly improved the model's performance.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to create a pull request or open an issue.
