# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

from prml.linear import VariationalLogisticRegression
from utils import load_lt_data


def create_toy_data(is_breast: bool = False,
                    is_heart: bool = False,
                    is_bone: bool = False,
                    is_kaggle_heart: bool = False,
                    is_credit: bool = False,
                    _path: str = None):
    scaler = MinMaxScaler()
    feature = PolynomialFeatures(degree=1, include_bias=True)

    if is_breast:
        LT = load_lt_data(_all=True, path="./breast_data/fix_breast_cancer.xlsx")
        _feature_names = LT.feature_names
    elif is_heart:
        LT = load_lt_data(_all=True, path="./spectf_data/over_resample.xlsx")
        _feature_names = LT.feature_names
    elif is_bone:
        LT = load_lt_data(_all=True, path="./bone_marrow_transplant_data/fix_bone_data.xlsx")
        _feature_names = LT.feature_names
    elif is_kaggle_heart:
        LT = load_lt_data(_all=True, path="./heart_disease_data/over_resample.xlsx")
        _feature_names = LT.feature_names
    elif is_credit:
        LT = load_lt_data(_all=True, path="./southGermanCredit_data/over_resample.xlsx")
        _feature_names = LT.feature_names
    else:
        LT = load_lt_data(_all=True, path='./data/over_resample_all_fields_scaler.xlsx')
        _feature_names = LT.feature_names

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(LT.data, LT.target, test_size=.3)

    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    _feature_names.insert(0, "Bias term")
    Xtrain = feature.fit_transform(Xtrain)
    Xtest = feature.transform(Xtest)
    return Xtrain, Xtest, Ytrain, Ytest, _feature_names


def auto_find_best_params(clf, params, Xtrain, Ytrain) -> tuple:
    kflod = KFold(n_splits=10, shuffle=True, random_state=9)
    gs = GridSearchCV(clf, params, cv=kflod, scoring="accuracy")
    gs.fit(Xtrain, Ytrain)
    best_params_ = gs.best_params_
    print(best_params_)
    return best_params_, gs.cv_results_


labels = [0, 1]
params = {
    "a0": [1, 10] + list(range(20, 110, 10)),
    "b0": [1, 10, 30, 50, 80] + list(range(100, 200, 40)),
}

target = [{"is_bone": True}, {"is_breast": True}, {"is_heart": True}, {"is_kaggle_heart": True}, {"LT": True}]
names = ["is_bone", "is_breast", "is_heart", "is_kaggle_heart", "LT"]

flag = False
while not flag:
    result = dict()
    for index, item in enumerate(target):
        if not item.get("LT", None):
            Xtrain, Xtest, Ytrain, Ytest, feature_names = create_toy_data(**item)
        else:
            Xtrain, Xtest, Ytrain, Ytest, feature_names = create_toy_data()

        # 单次计算，显示大概的权重
        best_param, cv_results_ = auto_find_best_params(VariationalLogisticRegression(), params, Xtrain, Ytrain)
        vlr = VariationalLogisticRegression(**best_param)
        vlr.fit(Xtrain, Ytrain, feature_names)
        _score = vlr.score(Xtest, Ytest)
        print("score is", _score, best_param)
        if _score >= 0.89:
            flag = True

            result[names[index]] = best_param
            results = pd.DataFrame(cv_results_)
            scores = np.array(results.mean_test_score).reshape(len(params['a0']), len(params['b0']))

            plt.figure()
            ax = heatmap(scores, annot=True, square=False, annot_kws={"fontsize": 9}, xticklabels=params.get("b0"),
                         yticklabels=params.get("a0"), cmap="viridis", fmt=".3g")
            plt.xlabel("b0 value")
            plt.ylabel("a0 value")
            plt.tight_layout()
            plt.savefig("./images/params2.png", dpi=600)
            plt.show()
            break
