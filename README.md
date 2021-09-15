# burakhakanakin
##############################################################
#  1. 2010-2011 UK müşterileri için 6 aylık CLTV prediction yapacağız.
##############################################################

#######################
# Verinin Hazırlanması (Data Preperation)
#######################

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#######################
# Verinin excel'den okunması
#######################

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()

#######################
# Veri Ön İşleme
#######################

df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

#######################
# Lifetime veri yapısının hazırlanması
#######################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

# değişkenlerin isimlendirilmesi
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# monetary değerinin satın alma başına ortalama kazanç olarak ifade edilmesi
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# monetary sıfırdan büyük olanların seçilmesi
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency'nin 1'den büyük olması gerekmektedir.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]


#######################
# BG/NBD Modelinin Kurulması
#######################

# pip install lifetimes
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#######################
# GAMMA-GAMMA Modelinin Kurulması
#######################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)

#######################
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
#######################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head()
cltv_final.sort_values(by="clv", ascending=False)[10:30]

#  2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplalım.
#  1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz edelim.


# Dikkat! Sıfırdan model kurulmasına gerek yoktur. Önceki soruda oluşturulan
# model üzerinden ilerlenebilir.


##############################################################
# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplıyoruz.
##############################################################
cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,  # months
                                    freq="W",  # T haftalık
                                    discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head()

cltv12 = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                     cltv_df['recency'],
                                     cltv_df['T'],
                                     cltv_df['monetary'],
                                     time=12,  # months
                                     freq="W",  # T haftalık
                                     discount_rate=0.01)

rfm_cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")
rfm_cltv12_final.head()

##############################################################
# 2. 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz. Fark var mı?
# Varsa sizce neden olabilir?
##############################################################
rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)



##############################################################
# Bazı yorunlamalar yapalım
##############################################################
# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# 2. CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır. Yorumlayınız.
# 3. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

##############################################################
# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
##############################################################
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # months
                                   freq="W",  # T haftalık
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

cltv_final["cltv_segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()


##############################################################
# 2. CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
##############################################################
cltv_final.groupby("cltv_segment").agg({"mean"})
