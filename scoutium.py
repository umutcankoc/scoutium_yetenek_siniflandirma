import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score,recall_score,roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc,rcParams

#########################################
# Veri Setini Okuma
#########################################

# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okuyalım.

df1 = pd.read_csv('scoutium_attributes.csv', sep= ";")
df2 = pd.read_csv('scoutium_potential_labels.csv', sep = ";")

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştirelim.

df = df1.merge(df2, how="left", on=["task_response_id", "match_id", "evaluator_id", "player_id"])

#########################################
# Değişken İşlemleri
#########################################

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldıralım.

df = df.drop(df[df["position_id"] == 1].index) # satırları sildik.

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldıralım.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df = df.drop(df[df["potential_label"] == "below_average"].index)

# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturalım. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız

pivot_tablo = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id", values="attribute_value")

# Adım 6: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayalım ve “attribute_id” sütunlarının isimlerini stringe çevirelim.

pivot_tablo = pivot_tablo.reset_index()


#########################################
# ENCODING İŞLEMLERİ
#########################################

# Adım 7: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade edelim.

pivot_tablo.columns = pivot_tablo.columns.astype(str)

#LABEL
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(pivot_tablo, "potential_label")

# Adım 8: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayalım.

def grab_col_names(dataframe, cat_th=10, car_th=20):

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

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(pivot_tablo)
num_cols = [col for col in num_cols if col != "player_id"]


# Adım 9: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayalım.

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(pivot_tablo), columns=pivot_tablo.columns)

########################################
# VERİ BÖLÜTLEME
########################################

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)


#########################################
# MODEL KURULUMU
#########################################

model = LGBMClassifier()
model.fit(X, y)

########################################
# 10 KATLI CROSS VALIDATION
########################################

cv_result = cross_validate(model, X, y, cv = 10, scoring=["accuracy", "f1", "roc_auc"])

cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result['test_roc_auc'].mean()

