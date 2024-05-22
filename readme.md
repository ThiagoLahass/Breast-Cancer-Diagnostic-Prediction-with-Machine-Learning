# Breast Cancer Diagnostic

The idea of this mini-project is to apply data processing, apply some algorithm models and compare the metrics of each one, and then choose which one has the best performance.

The dataset can be found in [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Steps:
1. **Prepare data for use**
    - Read
    - Clean
    - Separate
2. **Plot information to gain insights**
3. **Train a model**
4. **Check the results**

## Data Preparation
- Loaded the data and verified no empty values.
- Coded the diagnosis column: Malignant = 1, Benign = 0.
- Normalized the continuous values.

## Data Exploration
- Checked dataset balance and found it partially balanced.
- Analyzed the correlation between attributes using a heatmap.

## Data Splitting
- Used SMOTE to balance the dataset.
- Split the data into training and testing sets.

## Machine Learning Models Applied
### KNeighborsClassifier
- Applied K-Nearest Neighbors with cross-validation.
- Evaluated the model with accuracy, AUC score, and confusion matrix.

### DecisionTreeClassifier
- Applied Decision Tree Classifier with cross-validation.
- Evaluated the model with accuracy, AUC score, and confusion matrix.

## Results
- Compared the accuracy and AUC scores of both models.
- Plotted the ROC curves and confusion matrices to visualize performance.

## How to Use This Repository

### Prerequisites
Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ThiagoLahass/Breast-Cancer-Diagnostic.git
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

1. Ensure you have the dataset file `wdbc.data` in a `data` directory within the project.

2. Run the cells

## Dependencies

- pandas
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

If you enjoyed the content of this project, please give it a star!
