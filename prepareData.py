# -*- encoding: utf-8 -*-
import json
import os.path

import numpy as np
import pandas as pd
from numpy import float64
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class PrepareData:
    PREDICT_CHAR = ['PRE_1', 'PRE_2', 'PRE_3', 'PRE_4', 'PRE_5', 'PRE_6', 'PRE_7', 'PRE_8', 'PRE_9', 'PRE_10', 'PRE_11',
                    'PRE_12', 'PRE_13', 'PRE_14', 'PRE_15', 'PRE_16', 'PRE_17', 'PRE_18', 'PRE_19', 'PRE_20', 'PRE_21',
                    'PRE_22', 'PRE_23', 'PRE_24', 'PRE_25', 'PRE_26', 'PRE_27', 'PRE_28', 'PRE_29', 'PRE_30', 'PRE_31',
                    'PRE_32']

    #               # 'LYM#1', 'LYM1', 'RBC1', 'HCT1', 'NEUT#1', 'NEUT1', 'PT_Ratio1', 'APTT_Ratio1', 'TP1',
    # 'BNP', 'LYM#', 'LYM', 'RBC', 'HCT', 'NEUT#', 'NEUT']
    # 'PT_Ratio', 'TT', 'TP', '住院天数', '乙肝（1是，0否）', '丙肝（1是，0否）', '肝癌（1是，0否）',
    # '血吸虫肝硬化（1是，0否）',
    # '胆汁淤积性肝硬化（1是，0否）,',
    # '自免肝（1是，0否）', '原因不明肝硬化（1是，0否）', '酒精肝（1是，0否）', '其他（1是，0否）', '饮酒史（有1无0）',
    # '吸烟史', '脑血管疾病',
    # '高血压', '肝炎', '冠心病', '慢性肾衰', '慢支', '术中输血（否0.是1）', 'T', 'HR', 'SBP', 'DBP',
    # 'SP02']

    # 'SPO2', 'LYM#_minus',
    # 'LYM_minus', 'RBC_minus', 'HCT_minus',
    # 'NEUT#_minus', 'NEUT_minus', 'PT_Ratio_minus', 'TP_minus', "AST", "BloodLoss"]

    def __init__(self, filePath: str, is_delete: bool = True, is_delete_predict: bool = True,
                 is_add_column: bool = False, is_delete_minus_column: bool = True, include_list: list = None):
        self.df = pd.read_excel(filePath)

        if include_list:
            for i in self.df.columns:
                if i not in include_list:
                    del self.df[i]
            print("仅保留经过筛选的字段")
        else:
            if is_delete:
                self.remove_predict_char()
            if is_delete_predict:
                try:
                    del self.df["诊断"]
                except:
                    pass
                print("已删除经过SPSS预测的数据")
            if is_add_column:
                self.add_column(is_delete_minus_column)

        self.rules = dict()

    def add_column(self, is_delete_minus_column):
        """
        带1的为术前数据，不带1的为术后数据，1_1的是术前缺失值处理后的数据 _1是对术后缺失值处理的数据
        """

        before_list = list()
        after_list = list()
        for c in self.df.columns:
            if not c.endswith("1") and c + "_1" in self.df.columns:
                after_list.append((c, c + "_1"))

            if c.endswith("1") and c + "_1" in self.df.columns:
                before_list.append((c, c + "_1"))

        for c in before_list + after_list:
            del self.df[c[0]]

        self.df.rename(columns={v: k for k, v in before_list + after_list}, inplace=True)

        for c in self.df.columns:
            if (not c.endswith("1")) and (c + "1" in self.df.columns):
                self.df[c + "_minus"] = abs(self.df[c] - self.df[c + "1"])
                if is_delete_minus_column:
                    del self.df[c + "1"]
                    # del self.df[c]

        print("添加数据差值成功")

    def remove_predict_char(self):
        for i in self.PREDICT_CHAR:
            try:
                del self.df[i]
            except Exception:
                pass
        print("删除预测数据成功")

    @staticmethod
    def check_column_is_null(row) -> bool:
        if row[1].isnull().any():
            return True
        return False

    def format_reason(self, column_name: str, reason: str, data):
        self.df[column_name] = self.df[column_name].fillna(data)
        self.rules[f"{column_name}填充规则"] = {
            "reason": reason,
            "value": data
        }

    def fillNan(self) -> tuple:
        # 可以直接用 inplace= True 代替列循环
        for row in self.df.iteritems():
            try:
                if self.check_column_is_null(row):
                    column_name = row[0]
                    real_type = row[1].dtype

                    if column_name == "age":
                        media = self.df[column_name].median()
                        self.format_reason(column_name, "年龄取中位数较为合理", media)
                        continue
                    elif column_name == "HOD":
                        mode = self.df[column_name].mode().astype(real_type)
                        self.format_reason(column_name, "住院天数取众数较为合理", mode[0])
                        continue
                    elif column_name == "NTproBNP" or column_name == "RBC":
                        media = self.df[column_name].median()
                        self.format_reason(column_name,
                                           'NTproBNP取中位数较为合理' if column_name == "NTproBNP" else "RBC取中位数较为合理",
                                           media)
                    else:
                        mean = round(self.df[column_name].mean().astype(real_type), 2)
                        self.format_reason(column_name, f"{column_name}按照平均数填充较为合理", mean)
            except Exception as e:
                print(e, row[0])
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit_transform(self.df)

        rules_path = self.save_rules()
        target_file_path = self.save_file()
        return rules_path, target_file_path

    def save_file(self, name: str = "./data/fix_data_all_fields.xlsx"):
        self.df.to_excel(name, index=None)
        print("文件缺失数据已经补全")
        return name

    def save_rules(self, name: str = "rules.json"):
        path = f"./data/{name}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.rules, ensure_ascii=False))
        return path


class SklearnPrepareData:
    def __init__(self, filePath: str):
        self.df = pd.read_excel(filePath)
        del self.df["诊断"]
        self.rules = dict()

    @staticmethod
    def check_column_is_null(row) -> bool:
        if row[1].isnull().any():
            return True
        return False

    def knn_imputer(self, num: int = 10):
        imputer = KNNImputer(n_neighbors=num)

        for row in self.df.iteritems():
            if self.check_column_is_null(row):
                column_name = row[0]
                real_type = row[1].dtype

                if column_name == "诊断":
                    mapping = dict()
                    type_ = set()
                    for s in self.df[column_name]:
                        type_.add(s)

                    values = self.one_hot_covert(type_)

                    self.df[column_name] = self.df[column_name].replace(type_, values)
                    self.rules["诊断类型替换规则"] = {
                        "reason": "诊断类型非数值型，转换为数值类型-独热码",
                        "value": mapping
                    }
                    continue
                elif column_name == "年龄":
                    mode = self.df[column_name].mode().astype(real_type)
                    self.df[column_name] = self.df[column_name].fillna(mode[0])
                    self.rules["年龄填充规则"] = {
                        "reason": '年龄取众数较为合理',
                        "value": mode
                    }
                    continue
                elif column_name == "住院天数":
                    mode = self.df[column_name].mode().astype(real_type)
                    self.df[column_name] = self.df[column_name].fillna(mode[0])
                    self.rules["住院天数填充规则"] = {
                        "reason": '住院天数取众数较为合理',
                        "value": mode[0]
                    }
                    continue
                else:
                    imputer.fit_transform(self.df)


class PrepareBreastData:
    """
    未经过重采样
    """

    def __init__(self, path: str):
        """
        乳腺癌数据集中2为 温和 4为恶性
        """
        self.names = ["ID", "Clump_Thickness", 'Uniformity_of_Cell_Size', "Uniformity_of_Cell_Shape",
                      "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",
                      "Normal_Nucleoli", "Mitoses", "Class"]
        self.path = os.path.split(path)[0]
        self.df = pd.read_csv(path, header=None, names=self.names)

    def fillNan(self):
        del self.df["ID"]
        self.df.dropna(axis=0)
        self.df = self.df[~self.df.isin(['?'])].dropna(axis=0)

        # 将结果处理为0 1
        self.df["Class"].replace([2, 4], [0, 1], inplace=True)
        self.df.insert(0, "Class", self.df.pop("Class"))
        self.df.to_excel(os.path.join(self.path, "fix_breast_cancer.xlsx"), index=None)


class PrepareSpectData:
    """
    经过了重采样
    """

    def __init__(self, path: str):
        """
        uci spectf-heart心脏病数据集
        """
        self.names = ["Class", 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14',
                      'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22']
        self.path = os.path.split(path)[0]
        self.df = pd.read_csv(path, header=None, names=self.names)

    def fillNan(self):
        self.df.to_excel(os.path.join(self.path, "fix_spect_heart.xlsx"), index=None)


class PrepareBoneTData:
    """
    未经过重采样
    """

    def __init__(self, path: str):
        self.path = os.path.split(path)[0]
        self.df = pd.read_csv(path)

    def trans_one_hot(self):
        encoder = LabelEncoder()
        disease = encoder.fit_transform(self.df["Disease"].values)
        disease = np.array([disease]).T

        enc = OneHotEncoder()
        a = enc.fit_transform(disease)
        a = a.toarray()

        self.df = pd.concat([self.df, pd.DataFrame(a, columns=encoder.classes_)], axis=1)
        self.df = self.df.drop(['Disease'], axis=1)
        return encoder.classes_

    def fillNan(self):
        # 转为独热码以后将患有所有血液病的全置换为1
        classes_ = self.trans_one_hot()
        for i in classes_.tolist()[1:]:
            self.df.loc[self.df['ALL'] == 1, i] = 1

        self.df = self.df.replace(to_replace="?", value=np.nan)
        # 按列处理其他缺失数据
        for row in self.df.iteritems():
            column = row[0]
            real_type = row[1].dtype

            if row[1].isnull().any():
                mode = self.df[column].mode().astype(real_type)
                self.df[column] = self.df[column].fillna(mode[0])

        del self.df['ALL']
        self.df.insert(0, 'survival_status', self.df.pop('survival_status'))
        self.df.to_excel(os.path.join(self.path, "fix_bone_data.xlsx"), index=None)


class PrepareKaggleHeartData:
    """
    未经过重采样
    """

    def __init__(self, path: str):
        self.path = os.path.split(path)[0]
        self.df = pd.read_csv(path)

    def fillNan(self):
        self.df.insert(0, "target", self.df.pop("target"))
        self.df.to_excel(os.path.join(self.path, "over_resample.xlsx"), index=None)


class PrepareSouthGermanCreditData:
    def __init__(self, path: str):
        self.path = os.path.split(path)[0]
        self.df = pd.read_csv(path)

    def fillNan(self):
        self.df.insert(0, "kredit", self.df.pop("kredit"))
        self.df.to_excel(os.path.join(self.path, "over_resample.xlsx"), index=None)


if __name__ == '__main__':
    file_path = "./data/fix_data_all_fields.xlsx"
    prepare = PrepareData(file_path, is_add_column=True, is_delete_minus_column=True)
    prepare.fillNan()
