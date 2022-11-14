##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###################################
# İş Problemi (Business Problem)
###################################
# İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin
# gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

### Veri Seti Hikayesi ###
# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirket hediyelik eşyalar satmaktadır ve şirketin müşterilerinin büyük çoğunluğu toptancıdır.

### Değişkenler ###
# Invoice: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

##########################
# Gerekli Kütüphaneler
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#########################
# Verinin Okunması
#########################

df_ = pd.read_excel("online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()
df.info()

df.describe().T
df.head()
df.isnull().sum()

#########################
# Veri Ön İşleme
#########################

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

# bir ürüne ödediğim bedel
df["TotalPrice"] = df["Quantity"] * df["Price"]

#########################
# Plotting
#########################

df.dtypes
# Ülkelere göre alış-veriş sayıları
df.groupby('Country').Invoice.count().sort_values(ascending=False)

# En çok alış-veriş yapan ülkeler
plt.figure(figsize=(10, 10))
top5_sold_products = df.groupby('Country')['Invoice'].count().sort_values(ascending=False).head()
sns.barplot(x=top5_sold_products.index, y=top5_sold_products.values)
plt.xticks(rotation=80)
plt.xlabel('Country')
plt.ylabel('Shopping rates')
plt.title('Top 5 shopping countries')
plt.show()

df[df["Country"]=="Spain"]["StockCode"].count()

# En çok satılan ürünler
plt.figure(figsize=(10, 10))
top5_sold_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head()
sns.barplot(x=top5_sold_products.index, y=top5_sold_products.values)
plt.xticks(rotation=80)
plt.xlabel('Products')
plt.ylabel('Shopping rates')
plt.title('top 5 selling products')
plt.show()

# En çok alış-veriş yapan müşteriler ve ülkeleri
df.groupby(['Customer ID', 'Country']).Invoice.count().sort_values(ascending=False).head()

# En çok alış-veriş yapan müşteriler
plt.figure(figsize=(10, 10))
top5_sold_products = df.groupby('Customer ID')['Invoice'].count().sort_values(ascending=False).head()
sns.barplot(x=top5_sold_products.index, y=top5_sold_products.values)
plt.xticks(rotation=80)
plt.xlabel('Customer ID')
plt.ylabel('Shopping rates')
plt.title('top 5 shoppers')
plt.show()

# Ülkelere göre satılan ürünlerin dağılımı
df.groupby(['Description', 'Country']).Invoice.count().sort_values(ascending=False).head()


#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

last_date = df["InvoiceDate"].max()     #2011-12-09
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# işlem başına ortalama kazanç
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# müşterinin kaç haftadır alış-veriş yapmadığı
cltv_df["recency"] = cltv_df["recency"] / 7

# müşterinin kaç haftadır müşterimiz olduğu
cltv_df["T"] = cltv_df["T"] / 7


##############################################################
# Görev 1: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
##############################################################

# Adım 1: 2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,    # 6 months
                                   freq="W",  # frequency information of T
                                   discount_rate=0.01)


# Adım 2: Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz
cltv_merge = pd.merge(cltv_df, cltv, on="Customer ID", how="left")
cltv_merge.sort_values(by="clv", ascending=False).head(10)


##############################################################
# Görev 2:  Farklı Zaman Periyotlarından Oluşan CLTV Analizi
##############################################################

# Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
cltv_one_monts = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,    # 1 months
                                   freq="W",  # frequency information of T
                                   discount_rate=0.01)

cltv_merge["cltv_one_monts"] = cltv_one_monts
cltv_merge = pd.merge(cltv_merge, cltv_one_monts, on="Customer ID", how="left")

cltv_two_monts = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=2,    # 1 months
                                   freq="W",  # frequency information of T
                                   discount_rate=0.01)

cltv_merge["cltv_two_monts"] = cltv_two_monts
cltv_merge = pd.merge(cltv_merge, cltv_two_monts, on="Customer ID", how="left")

cltv_merge = pd.merge(cltv_merge, df[["Country", "Customer ID"]], on="Customer ID", how="left")

cltv_merge[cltv_merge["Country"] == "United Kingdom"][["cltv_one_monts", "cltv_two_monts", "Country"]]


# Adım 2: 1 aylık CLTV'den yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
cltv_merge[cltv_merge["Country"] == "United Kingdom"][["Customer ID", "Country", "cltv_one_monts"]].\
    sort_values(by="cltv_one_monts", ascending=False).head(10)

cltv_merge[cltv_merge["Country"] == "United Kingdom"][["Customer ID", "Country", "cltv_two_monts"]].\
    sort_values(by="cltv_two_monts", ascending=False).head(10)

# Adım 3: Fark var mı? Varsa sizce neden olabilir?
# cltv_one_monts: 16080.1149
# cltv_two_monts: 31880.3066


##############################################################
# Görev 3: Segmentasyon ve Aksiyon Önerileri
##############################################################

# Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız
# ve grup isimlerini veri setine ekleyiniz.
# clv_x: 6 aylık hesaplanan cltv değerleri
cltv_6_months_uk = cltv_merge[cltv_merge["Country"] == "United Kingdom"][["Customer ID", "Country", "clv_x"]]
cltv_6_months_uk["segment"] = pd.qcut(cltv_6_months_uk["clv_x"], 4, labels=["D", "C", "B", "A"])

# segmentlerin dağılımı
cltv_6_months_uk["segment"].value_counts()
# D    81635
# B    81179
# A    81100
# C    80804

sizes = [81635, 81179, 81100, 80804]
labels = 'D', 'B', 'A', 'C'
explode = (0.1, 0, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, autopct='%1.1f%%', shadow=True, startangle=75)
ax1.axis('equal')
ax1.set_title("Segments of customers by Cltv")
ax1.legend(labels)
plt.show()


# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
cltv_6_months_uk.groupby("segment").agg({"count", "mean", "sum"})

# clv değeri en yüksek olan 10 UK müşterisi
cltv_6_months_uk.sort_values(by="clv_x", ascending=False).head(10)











