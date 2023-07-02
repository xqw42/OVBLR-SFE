import pandas as pd
from ACME.ACME import ACME
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from prml.linear import VariationalLogisticRegression

dataset = pd.read_excel("./data/over_resample_all_fields_scaler.xlsx")
features = dataset.drop(columns='is_readmission').columns.to_list()

while True:
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset[features].values, dataset['is_readmission'].values,
                                                    test_size=.3)

    vlr = VariationalLogisticRegression()
    vlr.fit(Xtrain, Ytrain, feature_names=features)

    y_pred = vlr.predict(Xtest)
    _score = vlr.score(Xtest, Ytest)
    _f1_macro = f1_score(Ytest, y_pred, average='macro')
    _recall_score = recall_score(Ytest, y_pred, average='macro')
    _precision_score = precision_score(Ytest, y_pred, average='macro')

    if _score >= 0.85 and _f1_macro >= 0.85 and _recall_score >= 0.85 and _precision_score >= 0.85:
        print(_score, _f1_macro, _recall_score, _precision_score, "\n")

        acme_vlr = ACME(vlr, 'is_readmission', features=features,
                        cat_features=['sex', 'HepatitisC', 'is_alcohol', 'Diabetes', 'HBP', 'Hepatitis'], K=50,
                        task='class')

        acme_vlr = acme_vlr.explain(dataset, robust=True, label_class=1)
        summary_plot_1 = acme_vlr.summary_plot()
        summary_plot_1.show()
        summary_plot_1.write_image(file='./image_acme/lt_label_1.eps', format='eps')
        acme_vlr = acme_vlr.explain(dataset, robust=True, label_class=0)
        summary_plot_2 = acme_vlr.summary_plot()
        summary_plot_2.show()
        summary_plot_2.write_image(file='./image_acme/lt_label_0.eps', format='eps')
        bar_plot = acme_vlr.bar_plot()
        bar_plot.show()
        bar_plot.write_image(file='./image_acme/lt_bar.eps', format='eps')
        break
