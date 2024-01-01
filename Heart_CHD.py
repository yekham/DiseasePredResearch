import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_csv("Bitirme_Projesi/Datasets/framingham.csv")
filtered_data = df_heart[(df_heart['sysBP'] > 140) | ((df_heart['diaBP'] > 90) & (df_heart["prevalentHyp"] == 1))]
result = filtered_data['BPMeds']
result.value_counts()

report = sv.analyze(source=df_heart, target_feat="TenYearCHD" )
report.show_html("Bitirme_Projesi/HeartAttack_Analysis.html")

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
check_df(df_heart)


df_heart.isnull().sum()
#değişkenleri sınıflandıran fonksiyon
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
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car=grab_col_names(df,cat_th=10)

df.drop("education",axis=1,inplace=True)


cat_cols, num_cols, cat_but_car=grab_col_names(df,cat_th=10)

#aykırı değişkenler var ama göz ardı edilebilecek kadar az
#so we dont have to deal with outliers
def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
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
grab_outliers(df,"BPMeds")

for col in df.columns:
    print(col,grab_outliers(df,col))

df.info()
df.to_csv("Bitirme_Projesi/Datasets/framingham_NonNull.csv", index=False)
df=pd.read_csv("Bitirme_Projesi/Datasets/framingham_NonNull.csv",index_col=False)

filtered_data = df[(df['sysBP'] > 140) | ((df['diaBP'] > 90) & (df["prevalentHyp"] == 1))]
result = filtered_data['BPMeds']
result.value_counts()

df_heart.isnull().sum()
df.isnull().sum()
cat_cols, num_cols, cat_but_car=grab_col_names(df,cat_th=10)

df_heart.info()
df.iloc[14]
df.iloc[]
df_heart.iloc[14]
df.nunique()
df_heart.nunique()
df.dtypes
df_heart.dtypes
df.nunique()
#############--------------------VISUALIZATION------------#####################
#değişkenlerin korelasyonları
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
correlation_matrix(df,df.columns)


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
    target_summary_with_num(df, col, "TenYearCHD", plot=True)

def num_pairplot(dataframe,num_cols):
    sns.pairplot(dataframe[num_cols], kind='scatter',height=2.5)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()
num_pairplot(df,num_cols)

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
    target_summary_with_cat(df, 'TenYearCHD', col, plot=True)


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

########################################### MODELLEME AŞAMASI(PİPELİNE)############################################
import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df=pd.read_csv("Bitirme_Projesi/Datasets/framingham.csv")

X=df.drop("TenYearCHD",axis=1)
y=df["TenYearCHD"]


"""# StratifiedShuffleSplit sınıfını kullanarak veriyi bölme
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

# `n_splits=5` ile beş kez stratified sampling uygulanacak
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]"""


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.20)


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
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car=grab_col_names(X_train,cat_th=10)


def missing_value(dataframe):
    dataframe.drop(columns="education", inplace=True)

    BP_miss = dataframe[dataframe['BPMeds'].isnull()].index
    for i in BP_miss:
        if ((dataframe['sysBP'][i] > 140 or dataframe['diaBP'][i] > 90) and dataframe["prevalentHyp"][i] == 1):
            dataframe.loc[i, 'BPMeds'] = 1.0
        else:
            dataframe.loc[i, 'BPMeds'] = 0.0
    iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
    dataframe = pd.DataFrame(iterative_imputer.fit_transform(dataframe), columns=dataframe.columns).round(2)
    dataframe['BMI'] = dataframe['BMI'].round(1)

    for column in dataframe.columns:
        if column not in ['TenYearCHD', 'BMI','BPMeds']:
            dataframe[column] = dataframe[column].astype('int64')


    return dataframe

X_train=missing_value(X_train).set_index(X_train.index)

X_test=missing_value(X_test).set_index(X_test.index)




from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler

rs = RobustScaler()
X_train[num_cols] = rs.fit_transform(X_train[num_cols])
X_test[num_cols] = rs.transform(X_test[num_cols])

dump(rs, 'rs_CHD.joblib')


from imblearn.over_sampling import SMOTE, RandomOverSampler
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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, make_scorer
from sklearn.naive_bayes import GaussianNB
import joblib

#Confusion Matrix and Classification Report
def my_confusion_matrix(y_test, y_pred, plt_title):
    cm=confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='BuPu')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(plt_title)
    plt.show()
    return cm


################################################### BASE MODELS ###################################################
classifiers = {
    'Logistic Regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Support Vector Machines' : SVC(),
    'K-nearest Neighbors' : KNeighborsClassifier(),
    'XGBoost' : XGBClassifier(),
    'Gradient Boosting' : GradientBoostingClassifier(),
    'LightGBM' :  LGBMClassifier(),
    'AdaBoost' : AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(verbose=False)
}
results=pd.DataFrame(columns=['Accuracy in %','F1-score'])
for method,func in classifiers.items():
    func.fit(X_train_smote,y_train_smote)
    pred = func.predict(X_test)
    results.loc[method]= [100*np.round(accuracy_score(y_test,pred),decimals=4),
                         round(f1_score(y_test,pred),2)]
results



"""gradient_model=GradientBoostingClassifier()
gradient_model.fit(X_train_smote, y_train_smote)
y_predict_gradient=gradient_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_gradient,"GradientBoostingClassifier")


model=AdaBoostClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"AdaBoostClassifier")


model_log=LogisticRegression()
model_log.fit(X_train_smote, y_train_smote)
y_predict_log=model_log.predict(X_test)
my_confusion_matrix(y_test, y_predict_log,"LogisticRegression")


model_svc=SVC(probability=True)
model_svc.fit(X_train_smote, y_train_smote)
y_predict_svc=model_svc.predict(X_test)
my_confusion_matrix(y_test, y_predict_svc,"SVC")

model_xgb= XGBClassifier()
model_xgb.fit(X_train_smote, y_train_smote)
y_predict_xgb=model_xgb.predict(X_test)
my_confusion_matrix(y_test, y_predict_xgb,"XGBoost")

model=RandomForestClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"RandomForestClassifier")


model=KNeighborsClassifier()
model.fit(X_train_smote, y_train_smote)
y_predict=model.predict(X_test)
my_confusion_matrix(y_test, y_predict,"KNeighborsClassifier")"""

#####################################################################################
#XGBoost Optuna Optimization
#You can use optuna to find out the best hyperparameters for XGBoost.
#
def xgb_objective(trial):

    params = {
        "objective": 'binary:logistic',
        "eval_metric": 'logloss',
        "scale_pos_weight": trial.suggest_float('scale_pos_weight', 0.1, 1),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.3, 0.7),
        "subsample": trial.suggest_float('subsample', 0.3, 0.7),
        "learning_rate": trial.suggest_float('learning_rate', 0.001, 0.1),
        "max_depth": trial.suggest_int('max_depth', 1, 7),
        "gamma": trial.suggest_float('gamma', 0.001, 0.1),
        "min_child_weight": trial.suggest_int('min_child_weight', 1, 5),
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "use_label_encoder": False,
        "verbosity": 0
    }



    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=10, scoring='neg_log_loss')
    score = -scores.mean()

    return score


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(xgb_objective, n_trials=200)
xgb_best_params = study_xgb.best_params


xgb_best_params={'scale_pos_weight': 0.7520600861584015, 'colsample_bytree': 0.7282410339366315, 'subsample': 0.07757515062858956, 'learning_rate': 0.09169713049276583, 'max_depth': 8, 'gamma': 0.02485133175666666, 'min_child_weight': 1, 'n_estimators': 40}
xgb_best_params={'scale_pos_weight': 0.7886348060784316, 'subsample': 0.0976535440148534, 'learning_rate': 0.09954753007233802, 'max_depth': 6, 'gamma': 0.09180753791298944, 'min_child_weight': 3, 'n_estimators': 22}
xgb_best_params={'objective': 'binary:logistic','scale_pos_weight': 0.7886348060784316, 'colsample_bytree': 0.5768951506528086, 'gamma': 0.046887237712569516, 'learning_rate': 0.09818376221840938, 'max_depth': 6, 'min_child_weight': 1,  'n_estimators': 35, 'subsample': 0.8050309145863438}
xgb_best_params={'scale_pos_weight': 0.9992590745440851, 'colsample_bytree': 0.6723812608652413, 'subsample': 0.8170780442056526, 'learning_rate': 0.09533087439833346, 'max_depth': 10, 'gamma': 0.08751636831985661, 'min_child_weight': 1, 'n_estimators': 30}

xgb_model = XGBClassifier(**xgb_best_params)
xgb_model.fit(X_train_smote, y_train_smote)
y_predict_xgb=xgb_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_xgb,"XGBClassifier")

#####################################################################################


## Logistic Regression
"""def log_objective(trial):
    params = {
        'C': trial.suggest_loguniform('C', 0.0001, 1.0),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        "penalty": trial.suggest_categorical('penalty', ['l1', 'l2']),
        "solver": trial.suggest_categorical('solver', ['liblinear']),
    }
        #model = LogisticRegression(**params)
        #model.fit(X_train_smote, y_train_smote)
        #y_predict = model.predict(X_test)
        #score = log_loss(y_test, y_predict)

    model = LogisticRegression(**params)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=10, scoring='neg_log_loss')
    score = -scores.mean()

    return score


study_log = optuna.create_study(direction='minimize')
study_log.optimize(log_objective, n_trials=100)

log_best_params = study_log.best_params

log_model = LogisticRegression(**log_best_params)
log_model.fit(X_train_smote, y_train_smote)
y_predict_log=log_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_log,"LogisticRegression")"""

#AdaBoost Classifier
"""def ada_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8),
        'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
        'base_estimator' : trial.suggest_categorical('base_estimator',
                                                   [ None, 'deprecated'])
    }


    model = AdaBoostClassifier(**params)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=10, scoring='neg_log_loss')
    score = -scores.mean()
    return score


study_ada = optuna.create_study(direction='minimize')
study_ada.optimize(ada_objective, n_trials=100)

ada_best_params = study_ada.best_params

ada_model = AdaBoostClassifier(**ada_best_params)
ada_model.fit(X_train_smote, y_train_smote)
y_predict_ada=ada_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_ada,"AdaBoostClassifier")"""

#Gradient Boosting
"""def gradient_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth',1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 7)
    }


    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='neg_log_loss')
    score = -scores.mean()
    return score


study_gradient = optuna.create_study(direction='minimize')
study_gradient.optimize(gradient_objective, n_trials=50)

gradient_best_params = study_gradient.best_params
gradient_best_params = {'n_estimators': 100, 'learning_rate': 0.08161444334373921, 'max_depth': 5, 'min_samples_split': 3}
gradient_model = GradientBoostingClassifier(**gradient_best_params)
gradient_model.fit(X_train_smote, y_train_smote)
y_predict_gradient=gradient_model.predict(X_test)
my_confusion_matrix(y_test, y_predict_gradient,"GradientBoostingClassifier")"""




# Model Save
joblib.dump(xgb_model, 'xgb_model.joblib')

#Our Model for My Prediction Application
xgb_model = joblib.load('xgb_model.joblib')
params=xgb_model.get_params()
xgb_model_predict = xgb_model.predict(X_test)
my_confusion_matrix(y_test, xgb_model_predict,"XGBoost")



indices = y_test.loc[y_test == 1].index
print(indices)

rownum = 187
rowval = X_test.loc[rownum, :]
values_list = rowval.values.tolist()
print(values_list)

X_test_model = ([values_list])
xgb_model_predict_single = xgb_model.predict(X_test_model)



if (xgb_model_predict_single[0] == 1):
    lung_cancer = 'The person is having diabetes'
else:
    lung_cancer = 'The person does not have any diabetes'
print(lung_cancer)





#GridSearchCV Ensemble Learner

"""log_params = {
        'C': [0.0001, 1.0],
        'max_iter': [100, 500],
    }

xgboost_params = {
        "colsample_bytree": [0.1, 1.0] ,
        "subsample":  [0.1, 1.0],
        "learning_rate":  [0.01, 0.3],
        "max_depth": [1, 10],
        "gamma": [0.01, 1.0],
        "min_child_weight":  [1, 10],
        "min_child_samples": [1, 20],
        "n_estimators": [1, 300],
        "lambda": [0.1, 1.0],
        "alpha":  [0.1, 1.0]
                            }


classifiers = [
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                ('Logistic Regression', LogisticRegression(), log_params),
               ]



def hyperparameter_optimization(X, y, cv=10, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X_train_smote, y_train_smote)

######################################################
# 5. Stacking & Ensemble Learning
######################################################
#birden fazla modeli bir arada kullanmak
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]),
                                              ('Logistic Regression', best_models["Logistic Regression"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X_train_smote, y_train_smote)


y_predict = voting_clf.predict(X_test)
my_confusion_matrix(y_test, y_predict, 'Ensemble Model')
"""
























