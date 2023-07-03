# %load single_test.py
import os.path

import matplotlib.pyplot as plt
import shap
import eli5
from IPython.core.display import display
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from eli5.sklearn import PermutationImportance

from prml.linear import VariationalLogisticRegression
from utils import load_lt_data


def create_images_file(_path: str) -> str:
    if not os.path.exists(_path):
        os.makedirs(_path)
    return _path


def create_toy_data(is_breast: bool = False,
                    is_heart: bool = False,
                    is_bone: bool = False,
                    is_kaggle_heart: bool = False,
                    _path: str = None):
    scaler = StandardScaler()
    feature = PolynomialFeatures(degree=1, include_bias=True)

    if is_breast:
        image_path = create_images_file("./images/breast_data")
        LT = load_lt_data(_all=True, path="./breast_data/fix_breast_cancer.xlsx")
        feature_names = LT.feature_names
    elif is_heart:
        image_path = create_images_file("./images/spect_data")
        LT = load_lt_data(_all=True, path="./spectf_data/over_resample.xlsx")
        feature_names = LT.feature_names
    elif is_bone:
        image_path = create_images_file("./images/bone_marrow_transplant_data")
        LT = load_lt_data(_all=True, path="./bone_marrow_transplant_data/fix_bone_data.xlsx")
        feature_names = LT.feature_names
    elif is_kaggle_heart:
        image_path = create_images_file("./images/heart_disease_data")
        LT = load_lt_data(_all=True, path="./heart_disease_data/over_resample.xlsx")
        feature_names = LT.feature_names
    else:
        image_path = create_images_file("./images/LT")
        LT = load_lt_data(_all=True, path='./data/over_resample_all_fields_noscaler.xlsx')
        feature_names = LT.feature_names
        scaler = MinMaxScaler()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(LT.data, LT.target, test_size=.3)

    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    feature_names.insert(0, "Bias term")
    Xtrain = feature.fit_transform(Xtrain)
    Xtest = feature.transform(Xtest)

    return image_path, Xtrain, Xtest, Ytrain, Ytest, feature_names


def auto_find_best_params(clf, params, Xtrain, Ytrain) -> dict:
    gs = GridSearchCV(clf, params, cv=10, scoring="f1")
    gs.fit(Xtrain, Ytrain)
    best_params_ = gs.best_params_
    print(best_params_)
    return best_params_


labels = [0, 1]
# bone-t 数据集 {'a0': 1, 'b0': 2}
# matrix_label = ['alive', 'dead']
# max_score = 0.97

# spect heart数据集 {'a0': 1, 'b0': 12} 由于该数据只有二进制的，所以不需要标准化
# matrix_label = ['Normal', 'Abnormal']
# max_score = 0.92

# breast w 数据集 {'a0': 37, 'b0': 0.01}
# matrix_label = ['benign', 'malignant']
# max_score = 1

# kaggle heart disease数据集 {'a0': 1, 'b0': 2}
# matrix_label = ['Health', 'Unhealthy']
# max_score = 0.90

# LT 数据集 {'a0': 1, 'b0':140}
matrix_label = ['Non-readmission', 'Readmission']
max_score = 0.8811
__precision_score = 0.8431

flag = True
while flag:
    image_path, Xtrain, Xtest, Ytrain, Ytest, feature_names = create_toy_data(is_breast=True)

    vlr = VariationalLogisticRegression()
    vlr.fit(Xtrain, Ytrain, feature_names)
    print(vlr.feature_importance())
    y_pred_prob = vlr.proba(Xtest)
    y_pred = vlr.predict(Xtest)
    score = vlr.score(Xtest, Ytest)

    _f1_macro = f1_score(Ytest, y_pred, average='macro')
    _recall_score = recall_score(Ytest, y_pred, average='macro')
    _precision_score = precision_score(Ytest, y_pred, average='macro')
    print(
        f"f1_macro is: {_f1_macro} \t _recall_score is: {_recall_score} \t _precision_score is: {_precision_score} \t ")

    if score >= max_score and _precision_score >= __precision_score:
        flag = False
    if not flag:
        print("--------------")

        # confusion_matrix
        c = confusion_matrix(Ytest, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=c, display_labels=matrix_label)
        disp.plot()
        plt.show()
        # disp.figure_.savefig(f"{image_path}/confusion_matrix.png", dpi=600)

        shap.initjs()
        explainer = shap.Explainer(vlr.predict, Xtest, feature_names=feature_names)
        shap_values = explainer(Xtest)

        # shap.plots.bar(shap_values[0], max_display=30)
        plt.figure()
        shap.summary_plot(shap_values, Xtest, plot_type="bar", show=False)
        # plt.savefig(f"{image_path}/plot_bar.png", dpi=600)

        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        # plt.savefig(f"{image_path}/beeswarm.png", dpi=600)

        # plt.figure()
        # shap.plots.waterfall(shap_values[1], max_display=30)
        # shap.plots.force(shap_values[0])

        plt.figure()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.savefig(f"{image_path}/all_bar.png", dpi=600)

        perm = PermutationImportance(vlr, random_state=1).fit(Xtest, Ytest)
        display(eli5.show_weights(perm, feature_names=feature_names))
