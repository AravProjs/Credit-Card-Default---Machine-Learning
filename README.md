# Applied Machine Learning

## Default of Credit Card Clients - ML

### Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Data](#data)
- [Project Structure](#project-structure)
- [Results](#results)
- [Conclusion](#conclusion)
- [Further Improvements](#further-improvements)

### Overview
In this project, we apply machine learning techniques to solve real-world problems. The focus is on predicting whether a credit card client will default on their payment next month using various machine learning models. The analysis utilizes the "Default of Credit Card Clients Dataset" from [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset), which contains 30,000 examples and 24 features, including demographic information, credit data, and payment history. This dataset provides a foundation for analyzing factors influencing default risks, which can aid financial institutions in mitigating potential losses and making informed decisions.

### Objectives
- To understand the relationship between client features and credit card default.
- To implement, tune, and compare multiple machine learning models to predict defaults.
- To evaluate model performance using appropriate metrics and select the best-performing model.

### Data
The dataset used is the **Default of Credit Card Clients Dataset**, which contains:
- **30,000 samples** with **24 features**: demographic data, payment history, credit limit, etc.
- The target variable is `default.payment.next.month`, indicating whether a client will default.

### Project Structure

- **Data Splitting**: The dataset is split into a training set (70%) and a test set (30%) with a `random_state` of 76.
  - **Training set size**: 21,000 samples
  - **Test set size**: 9,000 samples
    
- **Exploratory Data Analysis (EDA)**: Initial analysis to understand data distribution and feature importance.
  - **Summary Statistic for `PAY_0`**  
    ![Summary Statistic for PAY_0](https://github.com/user-attachments/assets/ff5ccdb4-852b-4127-91f6-08f09112acb5)  
    The summary statistic for `PAY_0` reveals values like `-2` and `0` that lack a meaningful context in this problem. To address these irregularities, we imputed the values by replacing `-2` with `-1` and `0` with the mode. Similar issues were found across columns `PAY_0` to `PAY_6`, necessitating imputation for these variables.
  
  - **Gender Distribution**  
    ![Gender Distribution](https://github.com/user-attachments/assets/4be7ab9e-67d0-4cc9-9f60-195e15efbd22)  
    The data shows that the test subjects are predominantly female, which could impact the generalizability of the machine learning model across different genders.
  
  - **Age Distribution**  
    ![Age Distribution](https://github.com/user-attachments/assets/80f7fa44-e996-4e41-aeb2-335d16a75b60)  
    The histogram of the 'age' variable exhibits a right-skewed distribution, indicating a higher concentration of younger clients in the dataset. As age increases, the frequency decreases, which is typical for financial datasets where younger individuals are more numerous. Variations in certain age groups may suggest clusters of age-related behavior or demographic patterns important for credit risk assessment.
  
  - **Credit Limit by Education Level**  
    ![Credit Limit by Education Level](https://github.com/user-attachments/assets/bee8aba9-7dbb-485f-ad82-439488b0ff5a)  
    Analysis of credit limits by education level shows that clients with an education level of 2 have a broader range of credit limits, suggesting a diverse credit allocation within this group. The darkest hexbins (indicating the highest density of data points) for education levels 1 and 2 fall within the mid-range of credit limits, implying that most clients with these education levels receive moderate amounts of credit.

- **Preprocessing and Transformations**: Appropriate preprocessing techniques were applied to ensure the dataset is ready for machine learning models.
  - **Numerical Features**:  
    `LIMIT_BAL`, `AGE`, `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`  
    These features are continuous variables and may benefit from scaling to normalize their distribution, especially when using algorithms sensitive to the scale of data, such as logistic regression or support vector machines.

  - **Ordinal Features**:  
    `EDUCATION`, `PAY_0` to `PAY_6`  
    These features represent an ordinal scale:
    - **EDUCATION**: Ranges from 1 to 6, representing different education levels.
    - **PAY_0` to `PAY_6**: Represent repayment statuses from -1 to 9.

    While these features are already in a numerical format and reflect an order, it is important to ensure there are no missing or unexpected values. In the case of `PAY_0` to `PAY_6`, we observed some unexplained values (`-2`, `0`). We handled these by imputing them, where `-2` was replaced by `-1` and `0` was considered missing and imputed with the mode.

  - **One-Hot Encoded Features**:  
    `SEX`, `MARRIAGE`  
    These are nominal categories and were encoded using one-hot encoding to convert them into a format suitable for machine learning models. One-hot encoding is preferred over label encoding for nominal data because it does not imply an ordinal relationship between categories.

  ![Preprocessing Flow](https://github.com/user-attachments/assets/90ba1714-4fc3-4e3e-9ed2-b2b30cc91400)

- **Baseline Model**: Initial performance is evaluated using a baseline model (Dummy Classifier).
  ![Baseline Model Results](https://github.com/user-attachments/assets/b9c6851b-9110-43ac-aff1-d756992b6150)  
  Since there's more `0` (not default) value, the `DummyClassifier` naturally prioritized `0`. It's interesting to see similar test/train scores across folds, indicating that the class ratio is kept by the stratified K-Fold strategy during cross-validation.

- **Model Training**: Multiple models were trained, including logistic regression, decision trees, random forests, and histogram gradient boosting.
  - **Linear Pipeline**  
    ![Linear Pipeline Results](https://github.com/user-attachments/assets/51e5ea96-a60e-40fc-82a3-b31e8289b68b)  
    The validation score varies only slightly across different `C` values, with a low standard deviation, indicating that the scores are stable.

  - **Different Models**  
    ![Model Comparisons](https://github.com/user-attachments/assets/b509f6ca-d8be-44fd-b2df-c6ba714afd9e)  
    I initially wanted to run non-tree-based models like K-Nearest Neighbors (KNN) or Support Vector Machines (SVM), but they were too slow, causing my PC to crash from overheating.

    - **Decision Tree**: The decision tree model showed signs of overfitting, with a training score close to 1 and a validation score of around 0.71, indicating that it captures details specific to the training data that may not generalize well. However, it is the fastest model in terms of fit and score time.
    - **Histogram Gradient Boosting (HistGB)**: This model achieved the best results, outperforming the linear model without any significant overfitting or underfitting issues.
    - **Random Forest Classifier (RFC)**: The random forest model also exhibited overfitting similar to the decision tree. However, due to the ensemble's voting mechanism, its validation score was better than that of the linear model. The downside is its slower fit and score time compared to other models.

- **Hyperparameter Tuning**: Model hyperparameters were optimized using grid search and randomized search techniques to find the best parameter settings.
  ![Hyperparameter Tuning Results](https://github.com/user-attachments/assets/5bc4bd10-b7a9-4fde-8278-e20367c201d3)

- **Final Evaluation**: The best model was evaluated on the test set to assess its generalization performance.

### Results
- The **Histogram Gradient Boosting Classifier** emerged as the best-performing model, achieving an accuracy of **0.824** on the test set.
- **Decision Trees**: These models offer great performance and interpretability with a reasonably good validation score that does not exhibit overfitting, a common issue with this type of model. For this dataset, decision trees seem to be a good option.
- **Histogram Gradient Boosting (HistGB)**: Although it sacrifices some interpretability, this model delivers the best score, making it the top choice.
- **Random Forest**: Offers a balanced middle ground, performing slightly worse than HistGB but better than the decision tree. However, it suffers from longer fit times compared to all models.
- **Logistic Regression**: Provided a decent score but did not outperform the other models.

### Conclusion
All models tested are valid options for this dataset, but the **Histogram Gradient Boosting Classifier (HistGB)** stands out as the overall winner.

### Further Improvements
