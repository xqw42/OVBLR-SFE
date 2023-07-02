# -*- encoding: utf-8 -*-
import json
import random
from functools import lru_cache

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC

from prml.linear import VariationalLogisticRegression
from utils import load_lt_data, compute_cross_ten_test, draw_roc_curve


def get_data(path: str) -> object:
    LT = load_lt_data(_all=True, path=path)
    return LT


def computeCrossValue(clf, data, target, name: bool = False):
    _random = random.randint(0, 10)
    accuracy, precision, recall, f1, var_list, y_pred = 0, 0, 0, 0, list(), None
    for i in range(10):
        result = compute_cross_ten_test(clf, data, target, name)
        accuracy += result[0]
        precision += result[1]
        recall += result[2]
        f1 += result[3]
        var = result[4]
        var_list.append(var)

        if i == _random:
            y_pred = result[-1]

    # 2(Precision*Recall)/(Precision+Recall)
    accuracy_mean = (round(accuracy / 10, 4))
    precision_mean = (round(precision / 10, 4))
    recall_mean = (round(recall / 10, 4))
    f1_mean = (round(2 * (precision_mean * recall_mean) / (precision_mean + recall_mean), 4))
    processed_var = round(sum(var_list) / 10, 4)
    return accuracy_mean, precision_mean, recall_mean, f1_mean, var_list, processed_var, y_pred


def useLR(data, target):
    clf = LogisticRegression(solver="liblinear")
    return computeCrossValue(clf, data, target)


def useRF(data, target):
    clf = RandomForestClassifier()
    return computeCrossValue(clf, data, target)


def useKnn(data, target):
    clf = KNeighborsClassifier()
    return computeCrossValue(clf, data, target)


def useGN(data, target):
    clf = GaussianNB()
    return computeCrossValue(clf, data, target)


def useSVM(data, target):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    clf = SVC(probability=True)
    return computeCrossValue(clf, data, target)


def useGBC(data, target):
    clf = GradientBoostingClassifier()
    return computeCrossValue(clf, data, target)


def useADB(data, target):
    clf = AdaBoostClassifier()
    return computeCrossValue(clf, data, target)


@lru_cache()
def get_params() -> dict:
    params_list = [
        {'is_bone': {'a0': 1, 'b0': 12}, 'is_breast': {'a0': 27, 'b0': 27}, 'is_heart': {'a0': 12, 'b0': 12},
         'is_kaggle_heart': {'a0': 1, 'b0': 0.01}, 'LT': {'a0': 42, 'b0': 162}},
        {'is_bone': {'a0': 1, 'b0': 2}, 'is_breast': {'a0': 7, 'b0': 0.001},
         'is_heart': {'a0': 42, 'b0': 27},
         'is_kaggle_heart': {'a0': 22, 'b0': 0.1}, 'LT': {'a0': 1, 'b0': 45}},
        {'is_bone': {'a0': 1, 'b0': 17}, 'is_breast': {'a0': 17, 'b0': 0.001},
         'is_heart': {'a0': 17, 'b0': 2},
         'is_kaggle_heart': {'a0': 42, 'b0': 1}, 'LT': {'a0': 1, 'b0': 87}},
        {'is_bone': {'a0': 1, 'b0': 0.01}, 'is_breast': {'a0': 37, 'b0': 0.01},
         'is_heart': {'a0': 22, 'b0': 12}, 'is_kaggle_heart': {'a0': 42, 'b0': 7},
         'LT': {'a0': 1, 'b0': 142}},
        {'is_bone': {'a0': 1, 'b0': 1}, 'is_breast': {'a0': 27, 'b0': 0.01},
         'is_heart': {'a0': 1, 'b0': 0.01}, 'is_kaggle_heart': {'a0': 1, 'b0': 2},
         'LT': {'a0': 1, 'b0': 102}},
        {'is_bone': {'a0': 1, 'b0': 2}, 'is_breast': {'a0': 7, 'b0': 0.01},
         'is_heart': {'a0': 32, 'b0': 7},
         'is_kaggle_heart': {'a0': 12, 'b0': 0.01}, 'LT': {'a0': 1, 'b0': 147}},
        {'is_bone': {'a0': 1, 'b0': 22}, 'is_breast': {'a0': 7, 'b0': 0.01},
         'is_heart': {'a0': 1, 'b0': 12}, 'is_kaggle_heart': {'a0': 1, 'b0': 2},
         'LT': {'a0': 1, 'b0': 52}},
        {'is_bone': {'a0': 1, 'b0': 37}, 'is_breast': {'a0': 22, 'b0': 0.01},
         'is_heart': {'a0': 1, 'b0': 27},
         'is_kaggle_heart': {'a0': 1, 'b0': 0.01},
         'LT': {'a0': 1, 'b0': 147}},
        {'is_bone': {'a0': 1, 'b0': 7}, 'is_breast': {'a0': 7, 'b0': 0.1}, 'is_heart': {'a0': 1, 'b0': 1},
         'is_kaggle_heart': {'a0': 32, 'b0': 7}, 'LT': {'a0': 1, 'b0': 47}},
        {'is_bone': {'a0': 1, 'b0': 7}, 'is_breast': {'a0': 1, 'b0': 0.01},
         'is_heart': {'a0': 1, 'b0': 1}, 'is_kaggle_heart': {'a0': 1, 'b0': 2},
         'LT': {'a0': 1, 'b0': 157}}]

    params = {
        "Bone-marrow-T": list(),
        "Heart-disease": list(),
        "Spectf-heart": list(),
        "Breast-cancer": list(),
        "LT": list()
    }
    for i in params_list:
        for k, v in i.items():
            if k == "is_bone":
                params['Bone-marrow-T'].append(v)
            elif k == "is_breast":
                params['Breast-cancer'].append(v)
            elif k == "is_heart":
                params['Spectf-heart'].append(v)
            elif k == "is_kaggle_heart":
                params['Heart-disease'].append(v)
            else:
                params['LT'].append(v)

    params['Bone-marrow-T'] = [params.get("Bone-marrow-T")[1]]
    params['Breast-cancer'] = [params.get("Breast-cancer")[3]]
    params['Spectf-heart'] = [params.get("Spectf-heart")[6]]
    params['Heart-disease'] = [params.get("Heart-disease")[4]]
    params['LT'] = [params.get("LT")[3], {"a0": 1, 'b0': 42}, {"a0": 1, 'b0': 130}]
    return params


def useVLR(data, target, name, dataName: bool = False):
    target = target
    if name == "LT":
        pass
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)
    else:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    feature = PolynomialFeatures(degree=1, include_bias=True)
    data = feature.fit_transform(data)

    params = get_params()
    final_result, y_pred = list(), None
    _random = 0
    for index, param in enumerate(params.get(name)):
        clf = VariationalLogisticRegression(**param)
        *_result, _y_pred = computeCrossValue(clf, data, target, dataName)
        if index == _random:
            y_pred = _y_pred
        final_result.append(_result)

    return final_result, y_pred


if __name__ == '__main__':
    common_file_data = [
        "./bone_marrow_transplant_data/fix_bone_data.xlsx",
        "./heart_disease_data/over_resample.xlsx",
        "./spectf_data/over_resample.xlsx",
        "./breast_data/fix_breast_cancer.xlsx",
        "./data/over_resample_all_fields_scaler.xlsx"
    ]
    data_name = ["Bone-marrow-T", "Heart-disease", "Spectf-heart", "Breast-cancer", "LT"]

    result = dict()
    for name, path in zip(data_name, common_file_data):
        data_object = get_data(path)
        data = data_object.data

        result_dict = dict()
        *result_dict['LR'], y_lr_pred = useLR(data, data_object.target)
        *result_dict['KNN'], y_knn_pred = useKnn(data, data_object.target)
        *result_dict['SVM'], y_svm_pred = useSVM(data, data_object.target)
        *result_dict['VLR'], y_vlr_pred = useVLR(data, data_object.target, name,
                                                 dataName=False if name != "LT" else True)
        result[name] = result_dict

        if name == "Bone-marrow-T":
            draw_roc_curve(
                algorithm=("Logistic Regression", "K-Nearest Neighbor", "Support Vector Machine", "OVBLR-SFE"),
                prob_data=(y_lr_pred, y_knn_pred, y_svm_pred, y_vlr_pred), target=data_object.target,
                _path="./")
    with open("other_alg_result_all_fields_scaler_with_revised_99.json", "w") as f:
        f.write(json.dumps(result))
