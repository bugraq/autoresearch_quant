"""
YARIŞMACI FİLTRESİ testleri — para harcama güvenliği.

`compare.py` yarışmacı listesi iki AYRI soruyu cevaplayan iki grup içerir:

  (A) LLM'siz baseline'lar  -> "LLM gerçekten arıyor mu?"  (BEDAVA)
  (B) LLM'ler              -> "hangi model daha iyi?"     (~$2/koşu)

Eskiden hepsi zorunlu koşuyordu; bilimsel kontrolü (A) ölçmek isteyen para
harcamak zorundaydı. Muhtemelen bu yüzden baseline'lar bir noktada listeden
çıkarıldı ve projenin ANA iddiası ölçülemez duruma geldi.

Filtrenin en kritik kuralı ve bu dosyanın asıl sebebi:

    `cost` ETİKETİ YAZILMAMIŞSA ÜCRETLİ VARSAYILIR.

Yani hata yönü bilinçli seçilmiştir: fazladan bir yarışmacıyı atlamak,
habersiz API kredisi harcamaktan iyidir. Bu varsayılan ters çevrilirse
yeni eklenen etiketsiz bir model `--bedava` koşusunda sessizce para yakar.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare import _filtrele

_LISTE = [
    {"label": "random-search", "provider": "random", "cost": "free"},
    {"label": "genetic-programming", "provider": "gp", "cost": "free"},
    {"label": "bedava-llm", "provider": "openrouter", "cost": "free"},
    {"label": "pahali-llm", "provider": "openrouter", "cost": "paid"},
    {"label": "etiketsiz-llm", "provider": "openrouter"},      # cost YOK
]


def _etiketler(secili) -> list:
    return [c["label"] for c in secili]


def test_etiketsiz_yarismaci_UCRETLI_sayilir():
    """ASIL TEST: `cost` yazmayan model --bedava koşusunda KOŞMAMALI."""
    secili = _etiketler(_filtrele(_LISTE, None, bedava=True))
    assert "etiketsiz-llm" not in secili, (
        "cost etiketi olmayan yarışmacı BEDAVA sayıldı — yeni eklenen bir "
        "model habersiz API kredisi yakabilir")
    print("  [ok] cost etiketi olmayan yarışmacı ÜCRETLİ varsayılıyor")


def test_bedava_filtresi_ucretliyi_eler():
    secili = _etiketler(_filtrele(_LISTE, None, bedava=True))
    assert "pahali-llm" not in secili, "ücretli model --bedava'da koştu"
    assert set(secili) == {"random-search", "genetic-programming", "bedava-llm"}, secili
    print("  [ok] --bedava yalnız bedava yarışmacıları bırakıyor")


def test_sadece_tam_istenenleri_secer():
    secili = _etiketler(_filtrele(_LISTE, "random-search,pahali-llm", bedava=False))
    assert secili == ["random-search", "pahali-llm"], secili
    print("  [ok] --sadece tam istenen yarışmacıları seçiyor")


def test_sadece_bedavayi_EZER():
    """Açık istek örtük güvenliği ezer: kullanıcı adıyla istediyse koşar."""
    secili = _etiketler(_filtrele(_LISTE, "pahali-llm", bedava=True))
    assert secili == ["pahali-llm"], secili
    print("  [ok] --sadece açıkça istenen ücretliyi koşuyor (--bedava'ya rağmen)")


def test_bilinmeyen_etiket_SESSIZ_gecmez():
    """Yazım hatası, 'hiçbir filtre yok' gibi davranıp HEPSİNİ koşmamalı."""
    try:
        _filtrele(_LISTE, "randomsearch", bedava=False)   # tire eksik
    except SystemExit as e:
        assert "randomsearch" in str(e), str(e)
        assert "Mevcut" in str(e), "hangi etiketlerin var olduğu söylenmiyor"
        print("  [ok] bilinmeyen etiket hata veriyor (sessizce hepsini koşmuyor)")
        return
    raise AssertionError(
        "bilinmeyen etiket sessizce yutuldu — yazım hatası bütün listeyi "
        "koşturabilir (ücretli modeller dahil)")


def test_filtresiz_hepsi_gelir():
    secili = _filtrele(_LISTE, None, bedava=False)
    assert len(secili) == len(_LISTE)
    assert secili is not _LISTE, "aynı liste nesnesi döndü (yerinde değişme riski)"
    print("  [ok] filtresiz çağrı listeyi olduğu gibi (kopyalayarak) veriyor")


def test_config_baseline_iceriyor():
    """compare.yaml'da bilimsel kontrol GERÇEKTEN duruyor mu?

    Baseline'lar bir kez listeden çıkarıldı ve 'LLM random'dan iyi mi'
    sorusu ölçülemez oldu. Bu test onu geri çıkarmayı zorlaştırır.
    """
    import io

    import yaml

    yol = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "configs", "compare.yaml")
    with io.open(yol, encoding="utf-8") as f:
        comp = yaml.safe_load(f)["compare"]
    saglayicilar = {c.get("provider") for c in comp["contestants"]}
    eksik = {"random", "gp", "bayesopt"} - saglayicilar
    assert not eksik, (
        f"compare.yaml'da LLM'siz baseline eksik: {sorted(eksik)} — "
        f"'LLM gerçekten arıyor mu?' sorusu ölçülemez")
    # Her yarışmacı cost etiketli olmalı, yoksa --bedava onu sessizce atlar.
    etiketsiz = [c.get("label") for c in comp["contestants"] if "cost" not in c]
    assert not etiketsiz, f"cost etiketi olmayan yarışmacı(lar): {etiketsiz}"
    print(f"  [ok] compare.yaml {len(comp['contestants'])} yarışmacı, "
          f"3 baseline dahil, hepsi cost etiketli")


def main() -> None:
    test_etiketsiz_yarismaci_UCRETLI_sayilir()
    test_bedava_filtresi_ucretliyi_eler()
    test_sadece_tam_istenenleri_secer()
    test_sadece_bedavayi_EZER()
    test_bilinmeyen_etiket_SESSIZ_gecmez()
    test_filtresiz_hepsi_gelir()
    test_config_baseline_iceriyor()
    print("OK — yarışmacı filtresi güvenli (etiketsiz = ücretli) ve bilimsel "
          "kontrol config'de duruyor.")


if __name__ == "__main__":
    main()
