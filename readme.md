# Breast Cancer Diagnostic

The goal of this project is to develop a breast cancer diagnostic system by applying data processing techniques and comparing the performance of different machine learning models. The dataset used for this project is the [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Overview:
In this project, we follow several key steps:

1. **Data Preparation**
    - Load the dataset
    - Clean the data
    - Separate features and target variable
2. **Exploratory Data Analysis**
    - Visualize data to gain insights
    - Analyze correlations between features
3. **Data Preprocessing**
    - Encode categorical variables
    - Normalize continuous features
    - Balance the dataset using SMOTE
4. **Model Training**
    - Apply various machine learning algorithms:
        - `K-Nearest Neighbors (KNN)`
        - `Decision Tree`
        - `Support Vector Classifier (SVC)`
        - `Multinomial Naive Bayes`
        - `Gaussian Naive Bayes`
    - Evaluate each model using cross-validation and metrics like accuracy, AUC score, precision, recall, and F1-score
5. **Results and Model Selection**
    - Compare the performance of all models
    - Select the best-performing model based on key metrics

## Getting Started

### Prerequisites
Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ThiagoLahass/Breast-Cancer-Diagnostic-Prediction-with-Machine-Learning.git
    cd Breast-Cancer-Diagnostic
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On Unix or MacOS
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

1. Ensure you have the dataset file `wdbc.data` in the `data` directory.

2. Run the project by executing the cells in the Jupyter Notebook or Python script.

## Results and Model Selection
After training and evaluating multiple machine learning models, including `KNN`, `Decision Tree`, `SVC`, `Multinomial NB`, and `Gaussian NB`, we compared their performance based on key metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1-Score

We selected the best model based on `recall`, which is *crucial* for *cancer diagnosis* to *minimize false negatives*. The selected model demonstrated robust performance across multiple metrics, ensuring accurate and reliable predictions for early breast cancer detection and effective treatment.

For a detailed analysis and comparison of models, refer to the Jupyter Notebook or Python script in this repository.

## Dependencies

- pandas
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

If you enjoyed the content of this project or find it helpful, please give it a star!
