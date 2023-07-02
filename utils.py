# -*- encoding: utf-8 -*-
import heapq
import json
import os.path
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import Bunch
from tabulate import tabulate


class ConvertExcelToSKLearnDataSet:
    def __init__(self, path: str, is_breast: bool = False, is_heart: bool = False, is_bone: bool = False):
        self.is_breast = is_breast
        self.is_heart = is_heart
        self.is_bone = is_bone
        self.path = os.path.split(path)[0]
        self.df = pd.read_excel(path)
        self.df.to_csv(os.path.join(self.path, "raw_data.csv"), index=None)

    def convert_data_set(self, _all: bool, num):
        data_set = Bunch()

        data_set.df = self.df
        data_set.data = self._get_feature()
        data_set.target = self._get_target()
        data_set.DESCR = self._get_descr()
        data_set.feature_names = self._get_feature_names()
        data_set.target_names = self._get_target_names()

        return data_set

    def _get_feature(self):
        """
        获取数据集特征值
        :return:
        """
        if self.is_breast or self.is_bone:
            data_feature = self.df.iloc[:, :-1]
        else:
            data_feature = self.df.iloc[:, 1:]
        data_np = np.array(data_feature)
        return data_np

    def _get_target(self):
        """
        获取数据集目标值
        :return:
        """
        if self.is_breast:
            data_target = self.df.iloc[:, -1]
        else:
            data_target = self.df.iloc[:, 0]
        data_np = np.array(data_target)
        return data_np

    def _get_descr(self):
        """
        获取数据集描述
        :return:
        """
        if self.is_breast:
            text = "本数据集为Breast-Cancer数据，样本数量：{}；" \
                   "特征数量：{}；目标值数量：{}；无缺失数据" \
                   "".format(self.df.index.size, self.df.columns.size, 1)
        elif self.is_heart:
            text = "本数据集为SPETF-heart数据，样本数量：{}；" \
                   "特征数量：{}；目标值数量：{}；无缺失数据" \
                   "".format(self.df.index.size, self.df.columns.size, 1)
        else:
            text = "本数据集为ICU再入院数据，样本数量：{}；" \
                   "特征数量：{}；目标值数量：{}；无缺失数据" \
                   "".format(self.df.index.size, self.df.columns.size, 1)
        return text

    def _get_feature_names(self):
        """
        获取特征名字
        :return:
        """
        fnames = list(self.df.columns.values)

        if self.is_breast:
            fnames.pop(-1)
        else:
            fnames.pop(0)
        return fnames

    def _get_target_names(self):
        """
        获取目标值名称
        :return:
        """
        if self.is_breast:
            tnames = list(self.df.columns.values)[-1]
        else:
            tnames = list(self.df.columns.values)[0]
        return tnames


def load_lt_data(num: int = 30,
                 path: str = "/Users/wxq/Desktop/dealWithCode/data/select/fix_data.xlsx",
                 _all: bool = False,
                 is_breast: bool = False,
                 is_heart: bool = False,
                 is_bone: bool = False):
    LT = ConvertExcelToSKLearnDataSet(path, is_breast=is_breast, is_heart=is_heart, is_bone=is_bone)
    LT = LT.convert_data_set(_all=_all, num=num)
    return LT


def statistic_feature_importance_by_add(feature_names, feat_importance, num: int = 5):
    feature_result = dict()
    for k, v in zip(feature_names, feat_importance):
        if v:
            feature_result[k] = v

    counter = Counter(feature_result)
    return counter.most_common(num)


def getListMaxNumIndex(num_list, topk=10):
    max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))
    min_num_index = map(num_list.index, heapq.nsmallest(topk, num_list))
    return list(max_num_index), list(min_num_index)


def getListPValue(num_list):
    res = [index for index, i in enumerate(num_list) if i and (i < 0.05)]
    return res


def statistic_feature_total(path: tuple):
    res = list()
    for i in path:
        with open(i, "r", encoding="utf8") as f:
            try:
                content = f.read()
                res.extend(json.loads(content))
            except Exception as e:
                print(str(e), i)
    counter = Counter(res)
    return counter.most_common(5)


def statistic_feature_total_by_and(path: tuple):
    res = list()
    for i in path:
        with open(i, "r", encoding="utf8") as f:
            try:
                content = f.read()
                res.append(json.loads(content))
            except Exception as e:
                print(str(e), i)
    return list(set(res[0]).intersection(res[1], res[2]))


def generate_seaborn(path: str):
    if not os.path.exists(path):
        raise Exception("文件路径不存在")

    df = pd.read_excel(path)
    del df["是否二次入ICU"]
    new_df = df.corr()

    plt.figure(figsize=(100, 100))
    sns.heatmap(new_df, annot=True, linewidths=3)
    plt.tight_layout()
    plt.savefig("./data/heatmap.png")
    plt.show()  # 显示图片


# auto find best params
def gridSearchCv(model, parameters, scoring, data, target, cv: int = 5, n_jobs: int = -1):
    clf = GridSearchCV(cv=cv, estimator=model, param_grid=parameters, scoring=scoring, n_jobs=n_jobs)
    clf.fit(data, target)

    return clf.best_params_, clf.best_score_


def computeScore(Ytest, y_pred):
    _accuracy = accuracy_score(Ytest, y_pred)
    _recall_score = recall_score(Ytest, y_pred)
    _f1_score = f1_score(Ytest, y_pred)
    _precision_score = precision_score(Ytest, y_pred, pos_label=1)
    return ("_accuracy", _accuracy), ("_recall_score", _recall_score), ("_f1_score", _f1_score), (
        "precision_score", _precision_score)


def check_split_data(data, target, test_size: float = 0.3):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=test_size)
    return Xtrain, Xtest, Ytrain, Ytest


def random_over_sample(path: str, is_scaled: bool = True, delete_features: tuple = None):
    if not os.path.exists(path):
        raise Exception("path not exists")

    parent_path = os.path.split(path)[0]
    df = pd.read_excel(path)

    if delete_features:
        for i in delete_features:
            del df[i]

    # check corr
    new_df = df.corr()

    if is_scaled:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(df.iloc[:, 1:])
        data = np.insert(data, 0, np.array(df.iloc[:, 0]), axis=1)
        scaled_df = pd.DataFrame(data, index=df.index, columns=df.columns)
    else:
        scaled_df = df

    smote_tomek = SMOTETomek(random_state=0)
    x_resample, y_resample = smote_tomek.fit_resample(np.array(scaled_df.iloc[:, 1:]),
                                                      (np.array(df.iloc[:, 0])).astype(int))

    print(f"len of x_resample={len(x_resample)}")
    print(f"len of y_resample={len(y_resample)}")

    total_np_data = np.column_stack((y_resample[:, None], x_resample))
    total_pd_data = pd.DataFrame(total_np_data, columns=df.columns)

    result = pd.concat([scaled_df, total_pd_data], axis=0)
    if is_scaled:
        _path = os.path.join(parent_path, "over_resample_all_fields_scaler.xlsx")
    else:
        _path = os.path.join(parent_path, "over_resample_all_fields_noscaler.xlsx")
    result.to_excel(_path, index=None)
    return _path


def compute_cross_ten_test(model, data, target, name: bool = False):
    scoring = ["accuracy", "precision", "recall", "f1"]
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=7)

    results = model_selection.cross_validate(estimator=model,
                                             X=data,
                                             y=target,
                                             cv=kfold,
                                             scoring=scoring)

    y_pred = cross_val_predict(model, data, target, method="predict_proba", cv=kfold)
    return results["test_accuracy"].mean(), results["test_precision"].mean(), results["test_recall"].mean(), \
           results["test_f1"].mean(), np.std(results["test_accuracy"]), y_pred


def read_arrf(file):
    path = os.path.split(file)[0]
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
        # df = df.sort_values(by="Class", ascending=False)
        df.to_csv(os.path.join(path, "raw_data.csv"), index=None)
    return df


def draw_roc_curve(algorithm: tuple, prob_data: tuple, target, _path):
    evaluation = pd.DataFrame(index=algorithm, columns=['fpr', 'tpr', 'pre', 'rec', 'auc'])

    for name, e in zip(algorithm, prob_data):
        evaluation.loc[f'{name}', 'fpr'], evaluation.loc[f'{name}', 'tpr'], thresholds = metrics.roc_curve(
            target,
            e[:, 1])
        evaluation.loc[f'{name}', 'auc'] = metrics.auc(evaluation.loc[f'{name}', 'fpr'],
                                                       evaluation.loc[f"{name}", 'tpr'])
        evaluation.loc[f'{name}', 'pre'], evaluation.loc[
            f'{name}', 'rec'], thresholds = metrics.precision_recall_curve(
            target, e[:, 1])

    plt.figure(1)
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    temp_result = list()
    for i in range(len(algorithm)):
        fpr = evaluation.iloc[i]['fpr']
        tpr = evaluation.iloc[i]['tpr']
        auc = evaluation.iloc[i]['auc']
        temp_result.append(auc)
        plt.plot(fpr, tpr, lw=2, label=evaluation.index[i] + '+AUC' + ' (%0.4f)' % (auc))
    plt.legend(loc=4)
    plt.savefig(f'{_path}/ROC_Curve.png', dpi=300)
    plt.show()


def draw_table(data: list, num: int = 30, table=None):
    if table is None:
        table = ['No.', 'Weight', 'Feature']
    _data = []
    for i in range(num):
        _data.append([i + 1, data[i][-1], data[i][0]])

    _data.insert(0, table)
    result = tabulate(_data)
    print(result)


def compute_95_per(mean: float, std: float, ):
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    return Decimal(lower).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP), Decimal(upper).quantize(
        Decimal('0.0000'), rounding=ROUND_HALF_UP)


if __name__ == '__main__':
    pass
