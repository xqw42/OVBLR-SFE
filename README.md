# OVBLR-SFE : An Optimal Variational Bayesian Logistic Regression (OVBLR) model with a Salient Feature Estimation strategy.

OVBLR-SFE is a novel interpretable machine learning method that leverages feature importance to enhance
interpretability. This method incorporates variational inference and a Bayesian framework to approximate the posterior
probability distribution, and utilizes the estimated parameters of the posterior distribution as weights for regression
coefficients. Additionally, we have defined the concept of significant features based on a 95% confidence interval (
95%CI) to facilitate the selection of important features in high-dimensional datasets.

The code for OVBLR-SFE is implemented based on [PRML](https://github.com/ctgk/PRML)
and [scikit-learn](https://scikit-learn.org/stable/index.html).

# Key Features

* Feature Importance: OVBLR-SFE focuses on identifying and quantifying the importance of features within a dataset.
* Variational Inference: The method utilizes variational inference techniques to approximate the posterior probability
  distribution.
* Bayesian Framework: OVBLR-SFE adopts a Bayesian framework, allowing for a principled approach to modeling and
  inference.
* Weighted Regression Coefficients: The estimated parameters of the posterior probability distribution are employed as
  weights for the regression coefficients.
* Significance-based Feature Selection: The concept of significant features is defined based on a 95% confidence
  interval (95%CI) to aid in selecting important features in high-dimensional datasets.

# Requirements
```python
  pip = "*"
  sklearn = "*"
  ipykernel = {version = "*", index = "https://pypi.douban.com/simple"}
  ffmpeg = "*"
  matplotlib = "*"
  scikit-learn = "*"
  pandas = "*"
  numpy = "*"
  imblearn = "*"
  seaborn = "*"
  openpyxl = "*"
  polling = "*"
  socks = "*"
  lime = "*"
  shap = "*"
  eli5 = "*"
  ipython = "*"
  jupyter = "*"
  mglearn = "*"
  self-paced-ensemble = "*"
  tabulate = "*"
  pymoo = "*"
  ```
# Install
```python
    python setup.py install
```

# Usage
1. Import the OVBLR-SFE module

```python
    from prml.linear import VariationalLogisticRegression
```

2. Prepare your dataset and ensure it is in the appropriate format.

3. Train the OVBLR-SFE model:

```python
    vlr = VariationalLogisticRegression(a0=1, b0=1)
    vlr.fit(X_train, y_train, feature_names)
```

4. Obtain feature importance scores:
```python
    importance_scores = vlr.mapping
```

5. Prediction using trained models:
```python
    y_pred_prob = vlr.proba(Xtest)
    y_pred = vlr.predict(Xtest)
    score = vlr.score(Xtest, Ytest)
```


# How to use it ?
You can run single_test.py.
For more results see the images and image_acme folder.
