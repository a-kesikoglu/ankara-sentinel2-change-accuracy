# -*- coding: utf-8 -*-
"""Turkish figure labels: patches matplotlib Text.set_text so that every label drawn by figures.py is
translated (longest phrases first) and decimal points in numeric labels become commas."""
import re
import matplotlib.text as mtext

PAIRS = [
 # workflow boxes
 ('Sentinel-2 L2A surface reflectance\n2018 and 2025, SCL + s2cloudless masking',
  'Sentinel-2 L2A yüzey yansıması\n2018 ve 2025, SCL + s2cloudless maskeleme'),
 ('Spring and summer median composites\n10 bands, 6 indices, 3 texture layers',
  'İlkbahar ve yaz medyan kompozitleri\n10 bant, 6 indeks, 3 doku katmanı'),
 ('Dynamic World (2018, 2025)\nWorldCover 2021, GHSL built-up 2018', 'Dynamic World (2018, 2025)\nWorldCover 2021, GHSL yapılı alan 2018'),
 ('Stable consensus training pixels\n(agreement, probability, homogeneity,\nGHSL screening)',
  'Kararlı uzlaşı eğitim pikselleri\n(uyum, olasılık, homojenlik,\nGHSL elemesi)'),
 ('Random Forest trained on pooled\nsamples, applied to both years\n3 × 3 majority filter',
  'Birleştirilmiş örneklerle eğitilen\nRandom Forest, iki yıla uygulandı\n3 × 3 çoğunluk filtresi'),
 ('PCC and CVA-constrained PCC\n(Otsu threshold), MMU = 5 pixels\ntransition matrices',
  'PCC ve CVA kısıtlı PCC\n(Otsu eşiği), en küçük birim 5 piksel\ngeçiş matrisleri'),
 ('Stratified random sample\n(8 strata, n = 500), blind VHR\ninterpretation for both dates',
  'Tabakalı rastgele örneklem\n(8 tabaka, n = 500), iki tarih için\nkör VHR yorumlama'),
 ('Design-based estimators:\nOA, UA, PA, area-proportion error\nmatrices, areas with 95% CI',
  'Tasarım tabanlı kestiriciler:\nOA, UA, PA, alan oranlı hata\nmatrisleri, %95 GA ile alanlar'),
 ('Comparisons and robustness: RF vs SVM,\nfeature sets, Dynamic World PCC,\ncloud threshold, reference subsets',
  'Karşılaştırma ve sağlamlık: RF ve SVM,\nöznitelik setleri, Dynamic World PCC,\nbulut eşiği, referans alt kümeleri'),
 ('Area-adjusted estimates of land cover change and transitions with 95% confidence intervals',
  'Arazi örtüsü değişimi ve geçişlerinin %95 güven aralıklı alan düzeltmeli kestirimleri'),
 # panel titles
 ('(c) Study area, Sentinel-2 summer 2025 composite', '(c) Çalışma alanı, 2025 yazı Sentinel-2 kompoziti'),
 ('(a) Sampling strata and validation pixels', '(a) Tabakalar ve doğrulama pikselleri'),
 ('(b) Training locations by class', '(b) Sınıflara göre eğitim konumları'),
 ('(c) Spatial separation', '(c) Mekânsal ayrım'),
 ('Distance to nearest training location (m)', 'En yakın eğitim konumuna uzaklık (m)'),
 ('(c) Magnitude at validation pixels by reference transition', '(c) Referans geçişine göre doğrulama piksellerinde büyüklük'),
 ('(d) CVA-constrained PCC change', '(d) CVA kısıtlı PCC değişimi'),
 ('(a) Class area: map count vs estimate', '(a) Sınıf alanı: harita sayımı ve kestirim'),
 ('(b) Grouped permutation importance', '(b) Gruplanmış permütasyon önemi'),
 ('(a) Classifier and feature set', '(a) Sınıflandırıcı ve öznitelik seti'),
 ('Decrease in overall accuracy (pp)', 'Genel doğruluktaki azalma (yüzde puanı)'),
 ('Magnitude (reflectance units)', 'Büyüklük (yansıma birimi)'),
 ('(b) Magnitude histogram', '(b) Büyüklük histogramı'),
 ('(a) CVA magnitude', '(a) CVA büyüklüğü'),
 ("(a) User's accuracy", '(a) Kullanıcı doğruluğu'), ("(b) Producer's accuracy", '(b) Üretici doğruluğu'),
 ('(c) Change products', '(c) Değişim ürünleri'), ('(c) Reference estimate', '(c) Referans kestirimi'),
 ('RF land cover', 'RF arazi örtüsü'), ('(b) Ankara province', '(b) Ankara ili'), ('(c) PCC change', '(c) PCC değişimi'),
 ('Spring + summer observations', 'İlkbahar + yaz gözlemleri'), ('Transition area (ha)', 'Geçiş alanı (ha)'),
 ('Overall accuracy (%)', 'Genel doğruluk (%)'), ('Accuracy (%)', 'Doğruluk (%)'), ('Area (10³ ha)', 'Alan (10³ ha)'),
 ('Estimate ± 95% CI', 'Kestirim ± %95 GA'), ('Map pixel count', 'Harita piksel sayımı'),
 ('of pixels above', 'piksel eşiğin üstünde'), ('Otsu threshold', 'Otsu eşiği'),
 ('Easting (km)', 'Doğu (km)'), ('Northing (km)', 'Kuzey (km)'), ('WGS 84 / UTM zone 36N', 'WGS 84 / UTM 36K dilimi'),
 ('Ankara\ncentre', 'Ankara\nmerkezi'), ('Feature set', 'Öznitelik seti'), ('Training class', 'Eğitim sınıfı'),
 ('Validation pixels', 'Doğrulama pikseli'), ('Validation pixel', 'Doğrulama pikseli'), ('Strata', 'Tabakalar'),
 ('Pixels', 'Piksel'), ('Magnitude', 'Büyüklük'), ('median', 'medyan'),
 ('Reference: no change', 'Referans: değişim yok'), ('Reference: change', 'Referans: değişim'),
 ('PCC change without CVA support', 'CVA desteği olmayan PCC değişimi'), ('CPCC change to built-up', 'CPCC yapılı alana değişim'),
 ('CPCC other change', 'CPCC diğer değişim'), ('Change to bare/sparse', 'Çıplak/seyreğe değişim'),
 ('Change to built-up', 'Yapılı alana değişim'), ('Change to low vegetation', 'Alçak bitki örtüsüne değişim'),
 ('Built-up gain UA', 'Yapılı alan\nartışı UA'), ('Built-up gain PA', 'Yapılı alan\nartışı PA'),
 ('Built-up\ngain', 'Yapılı alan\nartışı'), ('Built-up gain', 'Yapılı alan artışı'), ('Change UA', 'Değişim UA'),
 ('Change PA', 'Değişim PA'), ('Binary OA', 'İkili OA'), ('Trajectory OA', 'Gidişat OA'),
 ('Net low vegetation', 'Net alçak bitki örtüsü'), ('Net bare/sparse', 'Net çıplak/seyrek'), ('Net built-up', 'Net yapılı alan'),
 ('Total change', 'Toplam değişim'), ('Other change', 'Diğer değişim'), ('No change', 'Değişim yok'), ('(b) Change', '(b) Değişim'),
 ('Stable water', 'Kararlı su'), ('Stable tree cover', 'Kararlı ağaç örtüsü'), ('Stable low vegetation', 'Kararlı alçak bitki örtüsü'),
 ('Stable built-up', 'Kararlı yapılı alan'), ('Stable bare/sparse', 'Kararlı çıplak/seyrek'),
 ('Yapracık and Bağlıca', 'Yapracık ve Bağlıca'), ('Etimesgut and Eryaman', 'Etimesgut ve Eryaman'), ('Northern Sincan', 'Kuzey Sincan'),
 ('Class in', 'Sınıf'), ('CPCC map', 'CPCC haritası'), ('PCC map', 'PCC haritası'), ('(a) PCC haritası', '(a) PCC haritası'),
 ('Spring bands', 'İlkbahar bantları'), ('Summer bands', 'Yaz bantları'), ('Spring indices', 'İlkbahar indeksleri'),
 ('Summer indices', 'Yaz indeksleri'), ('Texture', 'Doku'), ('summer true colour', 'yaz gerçek renk'),
 ('spring false colour', 'ilkbahar yanlış renk'), ('valid observations', 'geçerli gözlem'),
 ('Low vegetation', 'Alçak bitki örtüsü'), ('Low veg.', 'Alçak bitki'), ('Tree cover', 'Ağaç örtüsü'), ('Bare/sparse', 'Çıplak/seyrek'),
 ('Built-up', 'Yapılı alan'), ('Water', 'Su'), ('Tree', 'Ağaç'), ('Bare', 'Çıplak'),
 ('Low veg. to Built-up', 'Alçak bitkiden\nyapılı alana'), ('Low veg. to Bare/sparse', 'Alçak bitkiden\nçıplak/seyreğe'),
 ('Bare/sparse to Low veg.', 'Çıplak/seyrekten\nalçak bitkiye'), ('Bare/sparse to Built-up', 'Çıplak/seyrekten\nyapılı alana'),
 ('(no change)', '(değişim yok)'),
 # print-size layouts
 ('Sentinel-2 L2A surface\nreflectance, 2018 and 2025\nSCL + s2cloudless masking',
  'Sentinel-2 L2A yüzey\nyansıması, 2018 ve 2025\nSCL + s2cloudless maskeleme'),
 ('Spring and summer\nmedian composites\n10 bands, 6 indices,\n3 texture layers',
  'İlkbahar ve yaz\nmedyan kompozitleri\n10 bant, 6 indeks,\n3 doku katmanı'),
 ('Dynamic World (2018, 2025)\nWorldCover 2021\nGHSL built-up 2018', 'Dynamic World (2018, 2025)\nWorldCover 2021\nGHSL yapılı alan 2018'),
 ('Stable consensus\ntraining pixels (agreement,\nprobability, homogeneity,\nGHSL screening)',
  'Kararlı uzlaşı\neğitim pikselleri (uyum,\nolasılık, homojenlik,\nGHSL elemesi)'),
 ('Random Forest trained\non pooled samples and\napplied to both years\n3 × 3 majority filter',
  'Birleştirilmiş örneklerle\neğitilen ve iki yıla\nuygulanan Random Forest\n3 × 3 çoğunluk filtresi'),
 ('PCC and CVA-constrained\nPCC (Otsu threshold)\nMMU = 5 pixels\ntransition matrices',
  'PCC ve CVA kısıtlı\nPCC (Otsu eşiği)\nen küçük birim 5 piksel\ngeçiş matrisleri'),
 ('Stratified random sample\n(8 strata, n = 500)\nblind VHR interpretation\nfor both dates',
  'Tabakalı rastgele örneklem\n(8 tabaka, n = 500)\niki tarih için\nkör VHR yorumlama'),
 ('Design-based estimators:\nOA, UA, PA, area-proportion\nerror matrices, areas\nwith 95% CI',
  'Tasarım tabanlı kestiriciler:\nOA, UA, PA, alan oranlı\nhata matrisleri, %95 GA\nile alanlar'),
 ('Comparisons and robustness:\nRF vs SVM, feature sets,\nDynamic World PCC, cloud\nthreshold, reference subsets',
  'Karşılaştırma ve sağlamlık:\nRF ve SVM, öznitelik setleri,\nDynamic World PCC, bulut\neşiği, referans alt kümeleri'),
 ('Area-adjusted estimates of land cover change\nand transitions with 95% confidence intervals',
  'Arazi örtüsü değişimi ve geçişlerinin %95 güven\naralıklı alan düzeltmeli kestirimleri'),
 ('Low veg. to\nBuilt-up', 'Alçak bitkiden\nyapılı alana'), ('Low veg. to\nBare/sparse', 'Alçak bitkiden\nçıplak/seyreğe'),
 ('Bare/sparse to\nLow veg.', 'Çıplak/seyrekten\nalçak bitkiye'), ('Bare/sparse to\nBuilt-up', 'Çıplak/seyrekten\nyapılı alana'),
 ('Yapracık and\nBağlıca', 'Yapracık ve\nBağlıca'), ('Etimesgut and\nEryaman', 'Etimesgut ve\nEryaman'),
 ('Decrease in overall\naccuracy (pp)', 'Genel doğruluktaki\nazalma (yüzde puanı)'),
]
PAIRS.sort(key=lambda p: -len(p[0]))
NUMERIC = re.compile(r'^[\s−\-+]?\d+\.\d+$')
_orig = mtext.Text.set_text


def translate(s):
    if not isinstance(s, str) or not s:
        return s
    m = re.search(r'(\d+\.\d)% of pixels above', s)
    if m:
        s = s.replace(m.group(0), 'eşik üstü pikseller: %' + m.group(1).replace('.', ','))
    if re.match(r'^\d{1,3}(,\d{3})+$', s):
        return s.replace(',', '.')
    if NUMERIC.match(s):
        return s.replace('.', ',')
    out, i = [], 0
    # single pass, longest match first, so translated text is not translated again
    while i < len(s):
        for en, tr in PAIRS:
            if s.startswith(en, i):
                out.append(tr); i += len(en); break
        else:
            out.append(s[i]); i += 1
    t = ''.join(out)
    t = re.sub(r'(?<=\d)\.(?=\d)', ',', t) if re.search(r'Otsu|%|medyan', t) else t
    return t


def _set_text(self, s):
    return _orig(self, translate(s) if isinstance(s, str) else s)


mtext.Text.set_text = _set_text
