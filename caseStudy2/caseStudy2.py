#Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz.
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("car_crashes")
df.columns
df.info()
["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

#Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
#değişkenlerin isimlerinin sonuna "FLAG" yazınız.

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

#Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
#değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz

og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()
"""#Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
Görev 12: who değişkenini dataframe’den çıkarınız.
Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız."""


import numpy  as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#1
df = sns.load_dataset("titanic")
df.head()
df.shape

#2
df["sex"].value_counts()

#3
df.nunique()

#4
df["pclass"].unique()

#5
df[["pclass", "parch"]].nunique()
#6
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()
#7
df[df["embarked"] == "C"].head(10)
#8
df[df["embarked"] != "S"].head(10)
df[df["embarked"] != "S "]["embarked"].unique()
#9
df[(df["age"] < 30) & (df["sex"] == "female")].head()
#10
df[(df["fare"] > 500) | (df["age"] > 70)].head()
#11
df.isnull().sum()
#12
df.drop("who" , axis=1, inplace=True)
#13
type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()
#14
df["age"].fillna(df["age"].median(), inplace=True)
df.isnull().sum()
#15
df.groupby(["pclass", "sex"]).agg({"survived" : ["sum", "count", "mean"]})
#16

def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))
#fonk kullanmadan
df["age_flag"] = df["age"].apply(lambda x : 1 if x<30 else 0)

#17
df = sns.load_dataset("tips")
df.head()
df.shape

#18
df.groupby("time").agg({"total_bill":["sum", "min", "mean", "max"]})

#19
df.groupby(["day","time"]).agg({"total_bill":["sum", "min", "mean", "max"]})

#20
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                          "tip":["sum", "min", "max", "mean"]})

#21
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

#22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#23

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape