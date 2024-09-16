# Applied Machine Learning

## Default of Credit Card Clients - ML

### Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Data](#data)
- [Project Structure](#project-structure)
- [Results](#results)
- [Key Insights](#key-insights)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

### Overview
This project is part of the CPSC 330 course, where we apply machine learning techniques to solve real-world problems. The focus of this project is on predicting whether a credit card client will default on their payment next month using various machine learning models. The analysis utilizes the "Default of Credit Card Clients Dataset" from [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset), which contains 30,000 examples and 24 features, including demographic information, credit data, and payment history.

### Objectives
- To understand the relationship between client features and credit card default.
- To implement, tune, and compare multiple machine learning models to predict defaults.
- To evaluate model performance using appropriate metrics and select the best-performing model.

### Data
The dataset used is the **Default of Credit Card Clients Dataset**, which contains:
- **30,000 samples** with **24 features**: demographic data, payment history, credit limit, etc.
- The target variable is `default.payment.next.month`, indicating whether a client will default.

### Project Structure

### Project Structure
- **Data Splitting**: The dataset is split into a training set (70%) and a test set (30%) with a `random_state` of 76.
- **Exploratory Data Analysis (EDA)**: Initial analysis to understand data distribution and feature importance.
- **Preprocessing and Transformations**: Includes scaling, one-hot encoding, and handling missing values.
- **Baseline Model**: Initial performance is evaluated using a baseline model (Dummy Classifier).
- **Model Training**: Multiple models were trained, including logistic regression, decision trees, random forests, and histograms gradient boosting.
- **Hyperparameter Tuning**: Optimization of model hyperparameters using grid search and randomized search.
- **Final Evaluation**: The best model was evaluated on the test set to assess its generalization performance.
### Results
- The **Histogram Gradient Boosting Classifier** emerged as the best-performing model, achieving an accuracy of **0.824** on the test set.
- Other models like Decision Trees and Random Forests also showed good performance but with some overfitting issues.

### Key Insights
- The dataset had some class imbalance, influencing model evaluation metrics.
- The best-performing model was selected based on a combination of cross-validation accuracy, fit time, and model complexity.
- Feature engineering, including handling missing values and proper encoding, played a crucial role in improving model performance.

### Future Work
- Experiment with additional feature engineering techniques, such as creating interaction terms or applying PCA for dimensionality reduction.
- Explore other evaluation metrics, such as F1-score or recall, to better handle class imbalance.
- Implement more advanced hyperparameter optimization methods, such as Bayesian optimization.

### Installation
To run this project, you will need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
