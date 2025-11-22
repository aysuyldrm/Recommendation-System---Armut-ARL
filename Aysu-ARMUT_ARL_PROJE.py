
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.
import pandas as pd

#Görüntüleme ayarları
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# CSV dosyasını okuma ve bir DataFrame oluşturma
df=pd.read_csv(r"C:\Users\NEWUSER\Desktop\MIUUL\Recommendation_system\Armut\armut_data.csv")

# İlk 5 satıra göz atma
df.head()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head()

#Örnek:
#4 + 5 → Hizmet 4_5 → Koltuk Yıkama
#4 + 7 → Hizmet 4_7 → Petek Temizliği

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.


# 1. 'CreateDate' sütununu sadece yıl ve ay içerecek şekilde dönüştürelim
#type("CreateDate")
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["YearMonth"] = df["CreateDate"].dt.to_period("M")
df.head()

#CreateDate = 2023-07-15  →  YearMonth = 2023-07

# 2. UserId ve YearMonth bilgilerini birleştirerek sepet ID'si (fatura benzeri yapı) oluşturalım
df["SepetID"] = df["UserId"].astype(str) + "_" + df["YearMonth"].astype(str)
df.head()

df["CategoryId"].value_counts()
df1 = df[df["CategoryId"] == 4]
df1.shape
#Out[23]: (50631, 5)
df1["ServiceId"].value_counts()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

sepet_hizmet_df = df.groupby(["SepetID", "Hizmet"]).size().unstack().fillna(0).astype(int)
sepet_hizmet_df.head()

#her sepet (müşteri + ay) için hangi hizmetlerin alındığını gösteren bir tablo oluşturuyor.
#size() -> Her bir (SepetID, Hizmet) kombinasyonunun kaç kez geçtiğini sayıyor.
#unstack() -> Hizmet sütunu sütun isimlerine dönüştürülür — yani tablo geniş formata (pivot tablo) çevrilir.
#fillna(0) -> Boş (NaN) hücreleri 0 ile doldurur.

#Her satır bir Sepettir,
#Her sütun bir Hizmettir,
#Hücre değeri, o hizmetin sepette kaç kez geçtiğidir.

# Adım 2: Birliktelik kurallarını oluşturunuz.
#pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules

#Bu fonksiyonlar mlxtend (Machine Learning Extensions) adlı Python kütüphanesinden gelir.
#apriori() → Sık görülen hizmet kümelerini bulur. Örn: Sepetlerin %5’inde birlikte görülen hizmetler → sık hizmet kümeleri
#association_rules() → Bu kümelerden kurallar üretir. if, then... Örn: “Eğer 2_0 alınmışsa 15_1 alınma olasılığı 0.26”

# Sık görülen hizmet kümelerini çıkar

#sepet_hizmet_df içerisinde 1'den büyük değer var mı? 
#Eğer True dönerse → en az bir hücrede 1’den büyük değer var demektir.
(sepet_hizmet_df > 1).any().any()

#Kaç hücre etkileniyor?
(sepet_hizmet_df > 1).sum().sum()

#Yani: 26.525 hücrede bir hizmet aynı sepette birden fazla alınmış.

sepet_hizmet_df = sepet_hizmet_df.map(lambda x: 1 if x > 0 else 0)

#Toplam sepetlerin en az %1'inde birlikte görülen hizmetler
frequent_itemsets = apriori(sepet_hizmet_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

"""
#Örnek 1: support = 0.238121 , itemsets = (18_4)
Yani;
(18_4) hizmeti tüm sepetlerin %23,8’inde yer alıyor.
Yani bu hizmet çok popüler, neredeyse her 4-5 müşteriden biri almış.

#Örnek 2: support = 0.033951 , itemsets = (15_1, 2_0)
Bu artık bir kombinasyon (ikili hizmet seti).
Yani;
(15_1) ve (2_0) hizmetleri aynı sepette birlikte %3,4 oranında görülüyor.
“Müşterilerin yaklaşık %3,4’ü hem 15_1 hem de 2_0 hizmetlerini aynı ayda almış.”
"""

# Birliktelik kurallarını çıkar
# association_rules() fonksiyonu, apriori ile bulunan sık ürün kümelerinden birliktelik kurallarını oluşturur.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
#rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=3)

#filtered_rules = rules[(rules["support"]>0.01) & (rules["confidence"]>0.1) & (rules["lift"]>3)]
#filtered_rules


# En güçlü 5 kuralı göster
print(rules.sort_values("support", ascending=False).head())

"""
“Eğer bir müşteri X hizmetini aldıysa, **Y hizmetini alma olasılığı ne kadar?”
Ayrıca bu ilişki rastgeleliğe göre ne kadar güçlü?

antecedents:	Kuralın “öncül” kısmı → müşterinin aldığı hizmet
consequents:	Kuralın “sonuç” kısmı → o hizmetle birlikte alınan hizmet
support:    	Her iki hizmetin birlikte geçtiği sepetlerin oranı
confidence: 	A hizmeti alındığında, B’nin de alınma olasılığı
lift:       	Bu ilişki rastgeleliğe göre kaç kat daha güçlü
conviction: 	Kuralın ne kadar tutarlı olduğunu ölçer (1’e yakınsa zayıf, yüksekse güçlü)
zhangs_metric, jaccard, kulczynski	Alternatif ilişki gücü metrikleri (daha teknik ölçümler)
"""

"""
Örnek: 
Kural 1:
(2_0) → (15_1)
Support = 0.0339 → Sepetlerin %3,39’unda bu iki hizmet birlikte alınmış.
Confidence = 0.26 → 2_0 hizmetini alanların %26’sı 15_1 hizmetini de almış.
Lift = 2.15 → Bu ilişki rastgeleliğe göre 2,15 kat daha güçlü.

Yorum:
“2_0” hizmetini alan bir müşteriye “15_1” hizmetini önermek mantıklı, çünkü çoğunlukla birlikte satın alınıyorlar.
Bağlantılı hizmetler olabilir — “koltuk yıkama” ↔ “halı temizliği” gibi.
"""


#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
def arl_recommender(rules_df, product_id, rec_count=5):
    recommendation_list = []
    # 1. Boş bir liste oluşturuyoruz, buraya öneriler gelecek.

    #rules_df → Apriori ile oluşturulmuş birliktelik kuralları
    #product_id → Kullanıcının aldığı hizmet
    #rec_count → Kaç öneri istiyoruz

    # 2. Kuralları 'support' değerine göre büyükten küçüğe doğru sıralıyoruz.
    sorted_rules = rules_df.sort_values("support", ascending=False)

    # 3. Sıralanmış kuralların her birini teker teker incelemek için döngü başlatıyoruz.
    for _, rule in sorted_rules.iterrows():

        # 4. Eğer aradığımız ürün (product_id) bu kuralın öncüllerinde (antecedents) varsa:
        if product_id in rule["antecedents"]:

        # 5. Bu kuralın sonucunda önerilen ürünleri alıyoruz.
            recommended = next(iter(rule["consequents"]))

            # 6. Eğer bu öneri zaten listede yoksa, öneri listesine ekliyoruz.
            if recommended not in recommendation_list:
                recommendation_list.append(recommended)

            # 7. Eğer yeterince öneri bulduysak (örneğin 5 tane) döngüyü bitiriyoruz.
            if len(recommendation_list) >= rec_count:
                break

    # 8. Son olarak öneri listesini geri döndürüyoruz.
    return recommendation_list

# Fonksiyonu çağır ve tekrar edenleri ayıkla, sonuçları göster
recommended_services = arl_recommender(rules, "2_0", rec_count=5)
unique_recommendations = list(set(recommended_services))
print("2_0 hizmeti alan kullanıcıya önerilen hizmetler:", unique_recommendations)

recommended_services = arl_recommender(rules, "15_1", rec_count=5)
unique_recommendations = list(set(recommended_services))
print("15_1 hizmeti alan kullanıcıya önerilen hizmetler:", unique_recommendations)


recommended_services = arl_recommender(rules, "25_0", rec_count=5)
unique_recommendations = list(set(recommended_services))
print("25_0 hizmeti alan kullanıcıya önerilen hizmetler:", unique_recommendations)




filtered_rules = rules[
    (rules["support"] > 0.01) &
    (rules["confidence"] > 0.1) &
    (rules["lift"] > 1)
]

print(f"Tüm kurallar: {len(rules)} adet")
print(f"Filtrelenmiş (güçlü) kurallar: {len(filtered_rules)} adet\n")

def arl_recommender(rules_df, product_id, rec_count=5, fallback_rules=None):
    recommendation_list = []
    if rules_df.empty and fallback_rules is not None:
        print(" Filtrelenmiş kurallar bulunamadı. Tüm kurallar üzerinden öneri yapılıyor...")
        rules_df = fallback_rules
    sorted_rules = rules_df.sort_values("support", ascending=False)
    for _, rule in sorted_rules.iterrows():
        if product_id in rule["antecedents"]:
            recommended = next(iter(rule["consequents"]))
            if recommended not in recommendation_list:
                recommendation_list.append(recommended)
            if len(recommendation_list) >= rec_count:
                break
    return recommendation_list

recommended_services = arl_recommender(filtered_rules, "2_0", rec_count=5, fallback_rules=rules)
unique_recommendations = list(set(recommended_services))
print("2_0 hizmeti alan kullanıcıya önerilen hizmetler:", unique_recommendations)









