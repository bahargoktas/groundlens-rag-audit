# GroundLens — Derinlemesine Türkçe Çalışma Rehberi

## 1. Problem tanımı

Retrieval-Augmented Generation (RAG) sisteminde model cevap üretmeden önce bilgiyi bir doküman koleksiyonundan getirir. Bu projenin sınırını doğru bilmek önemli: GroundLens bir **LLM değil**, RAG sisteminin arama, kanıt seçimi ve atıf denetimi katmanını ölçen bir araştırma çalışmasıdır.

Temel problem: Soru ile alakalı **doğru kaynağı** ilk 1/5 sıraya getirmek, kaynak yetersiz olduğunda çekimser kalmak, seçilen cevap cümlesinin gerçekten gösterilen kaynakta yer almasını sağlamak.

### Veri

- `data/corpus.jsonl`: 55 kurgu belge, 47 aktif ve sekiz eski sürüm
- `data/queries.jsonl`: 52 soru; 18 geliştirme, 34 bağımsız test
- 24 cevaplanabilir test sorusu, 10 cevaplanamaz test sorusu

Bu belgeler gerçek şirket politikaları değildir. Bütün ölçümler yalnızca sentetik veri kümesine aittir.

## 2. Mimariyi takip et

`data.py` kurallara uygun JSONL okuyup belge ve soru nesneleri üretir. Belge durumlarını doğrular, benzersiz kimlikleri ve yalnızca aktif belgelere işaret eden gold etiketleri kontrol eder.

`retrieval.py`, üç farklı yöntemle puan üretir. Eski/superseded belgeler indeks dışında tutulur. Sonuç listesi her belge için kaynak kimliği, içerik, arama puanı ve ayrı bir sezgisel confidence değeri döndürür.

`answering.py`, arama sonuçlarındaki belgeleri confidence eşiğine göre gezer. Soru ile kelime örtüşmesi en yüksek olan özgün cümleyi seçer, aynı metnin kaynakta gerçekten olduğunu kontrol eder, eşik sağlanmazsa çekimser kalır.

`evaluation.py`, geliştirime bölümündeki etiketleri kullanarak eşiği seçer. Daha sonra test bölümünde Recall@k, MRR, nDCG, doğru kaynak atıf oranı ve cevaplanamaz soruda yanlış atıf oranını hesaplar. Test etiketlerini eşik optimizasyonunda kullanmaz.

`cli.py`, `groundlens evaluate` ve `groundlens ask` komutlarının giriş noktasıdır.

## 3. BM25 matematiği

BM25 bir sorgu teriminin belgede kaç kere geçtiğini, belge uzunluğunu ve terimin korpustaki nadirliğini dengeler.

```text
BM25(q,d) = sum_t IDF(t) *
            [f(t,d)*(k1+1)] /
            [f(t,d)+k1*(1-b+b*|d|/avgdl)]

IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t)+0.5))
```

Burada `f(t,d)` belgedeki terim frekansı, `df(t)` terimi içeren belge sayısı, `avgdl` ortalama belge uzunluğu, `k1=1.5`, `b=0.75` varsayılanlar.

**Örnek düşünce:** Çok belgede bulunan "policy" kelimesinin IDF değeri, nadir geçen "TLS" gibi bir teriminkinden düşüktür. Bu yüzden nadir terimler doğru kaynağı bulmada güçlü işaretler olabilir.

## 4. TF-IDF ve kosinüs

TF, terim frekansını; IDF, terimin belge koleksiyonunda ne kadar ayırt edici olduğunu temsil eder. `TfidfVectorizer` vektörleri normalize eder. Bir belge ile sorgu arasındaki iç çarpım, normalize edilmiş uzayda kosinüs benzerliğidir.

GroundLens iki ayrı TF-IDF matrisi öğrenir:
- Kelime 1-2 gramları; ağırlık 0.65
- `char_wb` karakter 3-5 gramları; ağırlık 0.35

Karakter n-gramları sözcüklerin farklı biçimlerinde arama eşleşmesine yardımcı olur. Sentaktik veya semantik anlamı bir dil modeli gibi öğrenmez.

## 5. Reciprocal Rank Fusion (RRF)

```text
RRF(d) = sum_{retriever} 1 / (k + rank_retriever(d))
```

Burada `k=60`. BM25 ve TF-IDF sıralamalarındaki üst sıralarda yer almak puanı artırır. RRF özgün ham skorların birbirinden farklı ölçeklerini kullanmaz. Bununla birlikte RRF'nin her veri kümesinde daha iyi olma garantisi yoktur. GroundLens'in sentetik kıyaslamasında TF-IDF tek başına daha güçlüdür.

## 6. Çekimser kalma ve atıf denetimi

Sistem sorgu–belge benzerliğinden ve normalize BM25 skorundan bir *heuristic confidence* hesaplar. Bu değer istatistiksel olarak kalibre edilmiş bir olasılık değildir.

Eşik, yalnızca 18 geliştime sorusunda seçilir. Hedef dev bölümünde cevaplanamaz soru üzerinde yanlış atıf oranını sınırlandırırken cevaplanabilir sorularda yeterli atıf üretmektir. Test skoruna bakarak eşiği değiştirmek veri sızıntısıdır.

Bir alıntının kaynakta kelimesi kelimesine bulunması **provenance** kanıtıdır. Bu, alıntının soruyu tam cevapladığı, doğru olduğu veya güvenilir olduğu anlamına gelmez.

Kodun `INSTRUCTION_RE` filtresi bazı açıkça komut gibi görünen metinleri ayıklar. Güvenlik politikası veya üretim düzeyi prompt-injection savunması değildir.

## 7. Değerlendirme metrikleri

```text
Recall@k = k içindeki gold kaynağı bulunan cevaplanabilir soru / cevaplanabilir soru sayısı
MRR@k    = (1/N) * sum_i [1 / ilk gold sıra_i], yoksa 0
DCG      = sum_i relevance_i / log2(i+1)
nDCG     = DCG / ideal DCG
```

Doğru kaynak atıf oranı: cevaplanabilir sorularda döndürülen atıf `gold evidence_id` ile aynı mı?

Yanlış atıf oranı: cevaplanamaz sorulardan kaçında atıf üretildi?

Şu ayrımı unutma: `Recall@5=1.0` olsa bile çekimser kalma nedeniyle *supported citation rate* daha düşük olabilir.

## 8. Çalıştırma

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m groundlens evaluate --out-dir reports_local
python -m groundlens ask "How often should live API keys be rotated?"
```

`reports_local/metrics.json` toplu skorları, `audit.jsonl` her test örneğinin atıf kararını, `REPORT.md` insan tarafından okunabilir özet raporu, `retrieval_comparison.png` iki erişim metriğinin grafiğini içerir.

## 9. Deneyler

1. `Retriever.rank` içinde TF-IDF ağırlıklarını 0.65/0.35 yerine 1/0 yap. Recall ve MRR değişiyor mu?
2. `reciprocal_rank_fusion` sabitini 60'tan 10'a indir. Sıralamalar nasıl değişiyor?
3. Bir belgeyi `superseded` olarak değiştir. Gold etiketlerinin neden güncellenmesi gerektiğini göster.
4. `--max-dev-fpr 0` ile dev eşik seçimini sıkılaştır. Testte desteklenen atıflara etkisini raporla.
5. 10 cevaplanamaz örneğin az olduğunu düşün. Daha çok ve benzer konuda yanlış-negatif soru eklemek nasıl bir fark yaratabilir?
6. Kaynaktaki cümle doğru görünse bile anlamsal olarak soruyu karşılamadığı örnekler tasarla.

## 10. Teknik mülakat: sorular ve kısa cevaplar

**1. Neden BM25?** Kelime bazlı güçlü, yorumlanabilir ve GPU gerektirmeyen bir baseline olduğu için.

**2. TF-IDF neden karakter n-gramı kullanıyor?** Yazım farklılıkları ve benzer alt dizeleri yakalamak için.

**3. RRF ile skor ortalaması arasındaki fark?** RRF sıralama pozisyonlarını birleştirir ve ham skor ölçeklerinden etkilenmez.

**4. Veri sızıntısı nasıl engellendi?** Eşik yalnızca dev etiketleriyle ayarlanıyor; holdout test optimize edilmiyor.

**5. Neden Gold source@5 cevap kalitesini ölçmez?** Bulunan doğru belge, üretilen metnin doğru veya alakalı olmasını garanti etmez.

**6. Confidence değeri neyi temsil ediyor?** Yalnızca sezgisel metin benzerliği; kalibre olasılık değil.

**7. Yanlış atıf oranı neden önemli?** Sistemin kanıt yokken uydurma kaynakla yanıt üretme eğilimini ölçüyor.

**8. Neden 0/10 yanlış atıf güvenlik kanıtı değil?** Örnek sayısı küçük ve veri sentetik.

**9. Prompt injection'a karşı güvende miyiz?** Hayır; regex filtresi ancak çok sınırlı örnekleri yakalar.

**10. Üretime geçmeden ne gerekir?** Gerçekçi bağımsız test setleri, semantik değerlendirme, güvenlik denetimi, erişim kontrolü, veri gizliliği ve izleme.

## 11. Öğrenme kontrolü

- BM25 formülündeki `k1` ve `b` parametreleri neyi kontrol eder?
- Bir belge sırası BM25'te 1, TF-IDF'te 3 ise RRF puanını hesapla.
- 24 cevaplanabilir sorunun 19'unda doğru atıf üretildiyse oran kaç?
- `false citation rate=0` değerinin klinik/gerçek dünya güvenliği hakkında neden kesin sonuç vermediğini açıkla.
- `evaluation.py` içindeki dev/test ayırımını teknik mülakatta kod üzerinden gösterebilir misin?
