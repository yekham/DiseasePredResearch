import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



df_diabetes=pd.read_excel("Bitirme_Projesi/Datasets/Diabetes.xlsx")
df_diabetes.info()

report = sv.analyze(source=df_diabetes )
report.show_html("Bitirme_Projesi/Diabetes.html")

def check_df(dataframe, head=5):
    """

    Args:
        dataframe: pandas dataframe
        head: görüntülenmek istenen satır sayısı

    Returns:

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)
check_df(df)


df.to_excel("Bitirme_Projesi/Datasets/diabetes_prep.xlsx", index=False)
df=pd.read_excel("Bitirme_Projesi/Datasets/diabetes_prep.xlsx")
df.Height.unique()

df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car=grab_col_names(df,cat_th=10)

def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in num_cols:
    print(col)
    grab_outliers(df, col,index=True)


#aykırı değişkenler en fazla Height'de var-> 3 adet
#aykırı değer problemimiz yok

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
correlation_matrix(df,num_cols)

def num_pairplot(dataframe,num_cols):
    sns.pairplot(dataframe[num_cols], kind='scatter',height=2.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
num_pairplot(df,num_cols)

def target_summary_with_num(dataframe, numerical_col, target_col, plot=False):
    if plot:
        plt.figure(figsize=(14, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(dataframe, x=numerical_col, hue=target_col, bins=20, palette='Set1', kde=True)
        plt.title(f"{numerical_col} Histogram")
        plt.xlabel(numerical_col)

        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=target_col, y=numerical_col, data=dataframe, palette='Set1')
        plt.title(f"{numerical_col} Box Plot")

        plt.tight_layout()
        plt.show()
for col in num_cols:
    target_summary_with_num(df, col, "Diabetes", plot=True)



def cat_summary(dataframe, col_name, plot=True):
    value_counts = dataframe[col_name].value_counts()
    value_ratios = 100 * value_counts / len(dataframe)

    if plot:
        plt.figure(figsize=(14, 6))

        # Count Plot
        plt.subplot(1, 3, 1)
        sns.set(style="whitegrid")
        sns.countplot(x=col_name, data=dataframe, palette="Set1", order=value_counts.index)
        plt.xticks(rotation=45)
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.title(f"{col_name} Count Plot")

        # Pie Chart
        plt.subplot(1, 3, 2)
        labels = value_counts.index
        sizes = value_counts.values
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title(f"{col_name} Pie Chart")

        plt.tight_layout()
        plt.show()
for col in cat_cols:
    cat_summary(df, col, True)

def target_summary_with_cat(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col+'_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)
for col in cat_cols:
    target_summary_with_cat(df, 'Diabetes', col, plot=True)



def FM_group(dataframe):
    #hastalığa sahip olan f ve m 'lerin yaş ortalaması
    grouped_data = dataframe[dataframe['TenYearCHD'] == 1].groupby(['male']).agg({'age': 'mean'}).reset_index()
    # Barplot oluşturma
    sns.barplot(x='male', y='age', data=grouped_data)
    # Grafik üzerine açıklama eklemek için
    plt.title('Yaş Ortalaması - TenYearCHD=1 ve Cinsiyet')
    plt.xlabel('Cinsiyet')
    plt.ylabel('Yaş Ortalaması')
    plt.show()
FM_group(dataframe=df)


############################################ MODELLEME AŞAMASI(PİPELİNE)############################################
import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_excel("Bitirme_Projesi/Datasets/Diabetes.xlsx")

X=df.drop("Diabetes",axis=1)
y=df["Diabetes"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20)


def X_data_prep(dataframe):
    cols = [i.replace(" ", "_") for i in dataframe.columns]
    dataframe.columns = cols
    dataframe.drop(columns="Patient_number",inplace=True)
    dataframe["Height"] = dataframe["Height"].apply(lambda x: x * 2.54).round(1)
    dataframe["Weight"] = dataframe["Weight"].apply(lambda x: x * 0.45359237).round(1)
    dataframe["BMI"] = (dataframe["Weight"] / (dataframe["Height"] / 100) ** 2).round(1)
    dataframe["waist"] = dataframe["waist"].apply(lambda x: x * 2.54).round(1)
    dataframe["hip"] = dataframe["hip"].apply(lambda x: x * 2.54).round(1)
    dataframe["Waist/hip_ratio"] = (dataframe["waist"] / dataframe["hip"]).round(2)
    dataframe["Gender"] = dataframe["Gender"].map({'female': 0, 'male': 1})
    dataframe['BMI'] = dataframe['BMI'].round(1)

    return dataframe

def Y_data_prep(dataframe):
    dataframe = dataframe.map({'No diabetes': 0, 'Diabetes': 1})
    return dataframe


X_train = X_data_prep(X_train)
y_train = Y_data_prep(y_train)

X_test = X_data_prep(X_test)
y_test = Y_data_prep(y_test)





def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car=grab_col_names(X_train,cat_th=10)
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler

rs = RobustScaler()
X_train[num_cols] = rs.fit_transform(X_train[num_cols])
X_test[num_cols] = rs.transform(X_test[num_cols])

dump(rs, 'rs_diabetes.joblib')

from imblearn.over_sampling import SMOTE
smote=SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


import warnings
warnings.simplefilter(action='ignore', category=Warning)


from collections import Counter
print("Before smote", Counter(y_train))
print("After smote", Counter(y_train_smote))


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report,f1_score,accuracy_score,ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.model_selection import cross_validate, cross_val_predict,cross_val_score
from sklearn import metrics
import optuna
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score
import joblib

def my_confusion_matrix(y_test, y_pred, plt_title):
    cm=confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return cm


classifiers = {
    'Logistic Regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Support Vector Machines' : SVC(),
    'K-nearest Neighbors' : KNeighborsClassifier(),
    'XGBoost' : XGBClassifier(),
    'Gradient Boosting' : GradientBoostingClassifier(),
    'LightGBM' :  LGBMClassifier(),
    'AdaBoost' : AdaBoostClassifier()
}
results=pd.DataFrame(columns=['Accuracy in %','F1-score'])
for method,func in classifiers.items():
    func.fit(X_train_smote,y_train_smote)
    pred = func.predict(X_test)
    results.loc[method]= [100*np.round(accuracy_score(y_test,pred),decimals=4),
                         round(f1_score(y_test,pred),2)]
results



"""model=GradientBoostingClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"GradientBoostingClassifier")


model=LGBMClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"LGBMClassifier")


model=SVC()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"SVC")

model=LogisticRegression()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"LogisticRegression")


model=XGBClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"LGBMClassifier")

model=XGBClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"XGBClassifier")"""






def lgbm_objective(trial):
    params = {
        "objective": 'binary',
        "metric": 'binary_logloss',
        "max_depth": trial.suggest_int('max_depth', 1, 10),
        "learning_rate": trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        "feature_fraction": trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        "bagging_freq": trial.suggest_int('bagging_freq', 1, 10),
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0)
    }

    # Veri setinizi ve hedef değişkeninizi kullanarak LightGBM modelinizi eğitin
    model = LGBMClassifier(**params)
    model.fit(X_train_smote, y_train_smote)
    y_predict = model.predict(X_test)
    score = log_loss(y_test, y_predict)

    return score


study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(lgbm_objective, n_trials=100)

lgbm_best_params = study_lgbm.best_params


lgbm_model = LGBMClassifier(**lgbm_best_params)
lgbm_model.fit(X_train_smote, y_train_smote)
y_predict_lgbm=lgbm_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_lgbm,"LGBMClassifier")

def xgb_objective(trial):
    params = {
        "objective": 'binary:logistic',
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.1, 1.0),
        "subsample": trial.suggest_float('subsample', 0.1, 1.0),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3),
        "max_depth": trial.suggest_int('max_depth', 1, 10),
        "gamma": trial.suggest_float('gamma', 0.01, 1.0),
        "min_child_weight": trial.suggest_int('min_child_weight', 1, 10),
        "min_child_samples": trial.suggest_int('min_child_samples', 1, 20),
        "n_estimators": trial.suggest_int('n_estimators', 1, 300),
        "booster": trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        "lambda": trial.suggest_loguniform("lambda", 0.1, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 0.1, 1.0),
        "verbosity": 0

    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=10, scoring='neg_log_loss')
    score = -scores.mean()
    return score


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=100)

xgb_best_params = study_xgb.best_params

xgb_model = XGBClassifier(**xgb_best_params)
xgb_model.fit(X_train_smote, y_train_smote)
y_predict_xgb=xgb_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_xgb,"XGBClassifier")

voting_classifier = VotingClassifier(estimators=[('lgbm', lgbm_model), ('xgb', xgb_model)], voting='soft')
voting_classifier.fit(X_train_smote, y_train_smote)
y_predict_voting = voting_classifier.predict(X_test)
my_confusion_matrix(y_test, y_predict_voting,"VotingClassifier")



# Modeli kaydet
joblib.dump(voting_classifier, 'ensemble_model.joblib')


ensemble_model = joblib.load('ensemble_model.joblib')
ensemble_model_predict = ensemble_model.predict(X_test)
my_confusion_matrix(y_test, ensemble_model_predict,"VotingClassifier")




indices = y_test.loc[y_test == 1].index
print(indices)

rownum = 311
rowval = X_test.loc[rownum, :]
values_list = rowval.values.tolist()
print(values_list)

X_test_model = ([[0.936585, 2.811881, 1.952381, -0.590909, 1.422680, 0,
                  -0.660131, -0.221080, 0.271777, 0.989691, 1.200000, 0.876847, -0.146893, 2.054054]])
ensemble_model_predict_arr = ensemble_model.predict(X_test_model)


if (ensemble_model_predict_arr[0] == 1):
    lung_cancer = 'The person is having diabetes'
else:
    lung_cancer = 'The person does not have any diabetes'
print(lung_cancer)





"""predicted_probabilities = voting_classifier.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds_roc = roc_curve(y_test, predicted_probabilities)
auc = roc_auc_score(y_test, predicted_probabilities)

precision, recall, thresholds_pr = precision_recall_curve(y_test, predicted_probabilities)

f1_scores = [f1_score(y_test, predicted_probabilities >= threshold) for threshold in thresholds_roc]

best_threshold_roc = thresholds_roc[np.argmax(f1_scores)]

print("En iyi eşik (ROC):", best_threshold_roc)"""


"""def svc_objective(trial):
    params = {
        "C": trial.suggest_loguniform('C', 0.001, 10.0),
        "gamma": trial.suggest_loguniform('gamma', 0.001, 10.0),
        "kernel": trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        "degree": trial.suggest_int('degree', 1, 10),
        "coef0": trial.suggest_int('coef0', 1, 10),
        "shrinking": trial.suggest_categorical('shrinking', [True, False]),
        "max_iter": trial.suggest_int('max_iter', 100, 1000),
        "class_weight": trial.suggest_categorical('class_weight', ['balanced', None]),
        'probability': True
    }
    model = SVC(**params)
    model.fit(X_train_smote, y_train_smote)
    y_predict = model.predict(X_test)
    score = log_loss(y_test, y_predict)
    return score


study_svc = optuna.create_study(direction='minimize')
study_svc.optimize(svc_objective, n_trials=100)

svc_best_params = study_svc.best_params

svc_model = SVC(**svc_best_params)
svc_model.fit(X_train_smote, y_train_smote)
y_predict_svc=svc_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_svc,"SVC")
"""

"""def log_objective(trial):
    params = {
        'C': trial.suggest_loguniform('C', 0.001, 10.0),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 10, 100),
        "penalty": 'l2',
        "solver": trial.suggest_categorical('solver', ['newton-cg','lbfgs'  ,'liblinear', 'saga']),
    }
    model = LogisticRegression(**params)
    model.fit(X_train_smote, y_train_smote)
    y_predict = model.predict(X_test)
    score = log_loss(y_test, y_predict)
    return score


study_log = optuna.create_study(direction='minimize')
study_log.optimize(log_objective, n_trials=100)

log_best_params = study_log.best_params

log_model = LogisticRegression(**log_best_params)
log_model.fit(X_train_smote, y_train_smote)
y_predict_log=log_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_log,"LogisticRegression")"""





