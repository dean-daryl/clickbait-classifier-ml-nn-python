## Introduction

With over 570 million internet users in Africa and rapidly growing digital media consumption, misleading clickbait content is becoming a major concern. In Rwanda and across the continent, it undermines media credibility, spreads misinformation, and wastes users' time. This project builds a machine learning model to automatically detect and flag clickbait headlines — supporting content moderation and promoting trustworthy digital communication in Africa.

Demo Video: https://www.loom.com/share/ecd93c824f1a4d74bce024b8843ed6bb

Dataset used: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset/data


### Model Comparison Table


| Instance | Optimizer | Regularizer          | Epochs | Early Stopping | Layers           | LR     | Accuracy | F1   | Recall | Precision |
|----------|-----------|-----------------------|--------|----------------|------------------|--------|----------|------|--------|-----------|
| 1  |Default(adam)  | None                  | 7      | No             | 64-32-16         | Default(0.001)| 0.96     | 0.95  | 0.93   | 0.95      |
| 2        | Adam      | L2 + Dropout          | 20     | Yes            | 128-64-32        | 0.001  | 0.94     | 0.94 | 0.92   | 0.96     |
| 3        | RMSprop   | L1 + Dropout          | 20     | Yes            | 128-64-32        | 0.001  | 0.91     | 0.91 | 0.89   | 0.94      |
| 4        | SGD    |    L1_L2 + Dropout       | 30     | Yes            | 128-64-32        | 0.01   | 0.89     | 0.88 | 0.85   | 0.92      |
| 5|Logistic Regression| N/A                   | N/A     | N/A           | N/A              | N/A    | 0.95     | 0.95 | 0.94   | 0.97      |


### Summary

### Which combination worked better

The baseline NN model in this case Instance 1 that was using the default Adam optimizer stood out even without regularization, dropout or early stopping. It achieved 0.95 f1 score with quite high recall(0.93) as well as precision(0.95). However the Logistic Regression performed equally well achieving f1 of 0.95 aswell. This indicates that this problem may be linearly separable with well-engineered TF-IDF features. By TF-IDF I mean Term Frequency-Inverse Document Frequency which is basically a feature extraction technique used to transform text into numerical vectors. It assigns high weight to words that are frequent in a specific document but rare accross the rest of the dataset. This helps you overlook common words in the headlines i.e("the", "and", "you") and focus on terms often used in clickbait like ("shocking", "you won't believe" ) and so on. This helped even a simple model like Logistic Regression become highly effective. Overall regularization did not impact NN performance but it contributed to preventing overfitting in deeper models.

### Which implementation worked better: ML Algorith or Neural Network

Both models did well in metrics like f1 score and recall, however the logistic regression was slightly better when it came to precision. Additionally it managed to train faster and effeciently. The logistic regression model used a `liblinear` Solver which is good for small datasets and it uses L2 regularization, C(Regularization) of `1.0` which is a moderate regularization(not too strong/weak) and `1000` maximum iterations for the solver to converge or in layman terms to make sure the model has been fully trained.


##  Running the Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/dean-daryl/clickbait-classifier-ml-nn-python.git
cd clickbait-classifier-ml-nn-python
```
### 2. Open the Notebook
```bash
  jupyter notebook notebook.ipynb
```

### 3. Run All Cells in Sequence

The notebook is modularized, with each model instance (Instance 1 to Instance 5) in its own section.

It includes data preprocessing, vectorization, model training, and evaluation.

Make sure the dataset file (clickbait_data.csv) is located in the project root or referenced correctly in the notebook.


### 4. Loading the best saved model

Here's a brief script to load the best saved model
```bash
  # Load the best saved model

logreg_model = joblib.load('saved_models/logreg_model.pkl')

# Predict
y_pred = logreg_model.predict(X_test)
````
