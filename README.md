# BGNBD-GG MODELİ İLE CLTV TAHMİNİ
##############################################################
#  1. We will make a CLTV prediction for 2010-2011 UK customers 6 month period 
#     2010-2011 UK müşterileri için 6 aylık CLTV prediction yapacağız.
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
# Verinin excel'den okunması (reading from excel)
#######################

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()

#######################
# Veri Ön İşleme (data preprocessing)
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
# Lifetime veri yapısının hazırlanması (preparing CLTV data strucure)
#######################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde)
#          Time since last purchase (weekly)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
#    Age of customer (weekly). How long ago from the analysis date the first purchase was made.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
#            total number of repeat purchases.
# monetary_value: satın alma başına ortalama kazanç
#                 average earnings per purchase
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

# değişkenlerin isimlendirilmesi (naming variables)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# monetary değerinin satın alma başına ortalama kazanç olarak ifade edilmesi ( Expressing monetary value as average earnings per purchase)
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# monetary sıfırdan büyük olanların seçilmesi (Choosing those greater than monetary zero)
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi (Expression of recency and T for BGNBD in weekly terms)
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency'nin 1'den büyük olması gerekmektedir (frequency must be greater than 1)
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]


#######################
# BG/NBD Modelinin Kurulması (Establishing the BG/NBD Model)
#######################

# pip install lifetimes
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#######################
# GAMMA-GAMMA Modelinin Kurulması (Establishing the GAMMA-GAMMA Model)
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
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması (Calculation of CLTV with BG-NBD and GG model)
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

#  2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplalım (Let's calculate 1-month and 12-month CLTV for 2010-2011 UK customers)
#  1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz edelim (Let's analyze the top 10 people at 1-month CLTV and the top 10 people at 12 months.)


##############################################################
# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplıyoruz (We calculate 1-month and 12-month CLTV for 2010-2011 UK customers)
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
#  1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz. Fark var mı? (Analyze the top 10 people at 1 month CLTV and the 10 highest at 12 months. Is there a difference?)
# Varsa sizce neden olabilir? (If so, why do you think it could be?)
##############################################################
rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)

##############################################################
#  2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. (For 2010-2011 UK customers, divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the dataset)
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


