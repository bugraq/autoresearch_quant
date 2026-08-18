"""
Model karşılaştırma koşucusu (Doküman 15/26 — "hangi LLM daha iyi arıyor?").

Aynı kampanya kısıtları + aynı veri + aynı deney bütçesi altında birden çok
hipotez üreticisini yarıştırır ve ARAŞTIRMA VERİMLİLİĞİ tablosu üretir.
Ölçülen şey "en iyi Sharpe" DEĞİL (Doküman 26): yapısal isabet oranı, keşif
hızı, kabul/tekrar oranı, çoklu-test sonrası en iyi DSR, token maliyeti.

configs/compare.yaml iki AYRI soruyu cevaplayan iki grup içerir:
    (A) LLM'siz baseline'lar (random-search / GP / bayes-opt) — BEDAVA
        "LLM gerçekten arıyor mu, rastgeleden iyi mi?" (Deney A)
    (B) LLM'ler — ~$2/koşu
        "hangi model daha iyi hipotez üretiyor?" (model seçimi)

Kullanım:
    python compare.py                       # hepsi (ücretli modeller DAHİL)
    python compare.py --bedava              # yalnız (A) + ücretsiz modeller
    python compare.py --sadece random-search,deepseek-r1-GUCLU-UCUZ
    python compare.py --seeds 3             # tek seed (hızlı deneme)

TEK KOŞU KURALI: aynı anda iki karşılaştırma koşulamaz (runs/.compare.lock).
Yarışmacı hafızaları etiket+seed'den türer ve her yarışmacıda silinip yeniden
kurulur; iki koşu üst üste binerse biri diğerinin veritabanını ortasında siler
ve sonuç ÇÖKME değil KARIŞMIŞ ÖLÇÜM olur (canlı yaşandı).

Adalet kuralları:
  - Her yarışmacıya AYRI, TAZE hafıza (runs/compare_<label>.sqlite).
  - Critic varsayılan dummy (deterministik) — tek değişken üretici olsun.
  - Literatür araması kapalı (koşular arası varyans katmasın).
  - Holdout'a ASLA dokunulmaz; karşılaştırma araştırma dönemi metrikleriyledir.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from statistics import mean

from dotenv import load_dotenv

# Windows konsolu (cp1254) LLM/rapor metnindeki ok/em-dash gibi karakterlerde
# UnicodeEncodeError ile PATLAR. main.py'deki korumanin aynisi (bkz. main.main).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from evaluation import build_report
from evaluation.improvement import best_so_far, improvement_svg
from llm import make_provider
from memory import MemoryStore
from orchestrator import run_campaign
from main import HERE, build_config, load_data, load_yaml


def _spec_ops_fields(hyp_json: str) -> "tuple[set, set, str]":
    """hypothesis_json -> (operatörler, veri alanları, model tipi)."""
    ops: set = set()
    fields: set = set()

    def walk(n):
        if isinstance(n, dict):
            op = n.get("op")
            if op:
                ops.add(op)
                if op == "field" and n.get("field"):
                    fields.add(n["field"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    h = json.loads(hyp_json)
    walk(h.get("signal", {}))
    for f in h.get("features", []):
        walk(f.get("expression", {}))
    return ops, fields, str((h.get("model") or {}).get("type", "dsl_formula"))


def _yapisal_isabet(ops: set, fields: set, model_tipi: str,
                    need_ops: set, need_any: set) -> bool:
    """Bu hipotez GERÇEK alpha'nın ailesini deniyor mu?

    ★ MODEL TİPİNE GÖRE FARKLI SORULUR — yoksa ölçüt geçersizdir.

    Ölçülen hedef "momentum × hacim ETKİLEŞİMİ"dir. Bunu denemenin iki
    meşru yolu var ve ikisi FARKLI görünür:

      dsl_formula : etkileşimi ELLE kurmak gerekir -> `multiply` düğümü ŞART.
      ML (random_forest / gbm / ...) : model etkileşimi KENDİSİ öğrenir;
          araştırmacının işi iki bacağı FEATURE olarak BESLEMEKtir. Açıkça
          çarpmak gereksizdir, hatta ML modunda anti-desendir.

    Bu ayrım yapılmadan ölçüldüğünde (canlı yaşandı, 30.07.2026) sonuç
    tamamen yanıltıcıydı:
        baseline'lar  : %100 dsl_formula  -> `multiply` yazmak zorunda
        LLM           : %100 random_forest -> hiç `multiply` yazmıyor
        LLM isabeti   : %0   ("hiç bulamadı")
    Oysa LLM'in 15 hipotezinin HEPSİ doğru malzemeyi veriyordu: close
    (return -> momentum) VE volume/dollar_volume. Yani ölçüt, LLM'i
    kullanmak ZORUNDA olduğu model tipi yüzünden cezalandırıyordu; ortaya
    çıkan "LLM rastgeleden kötü" tablosu bir ÖLÇÜM hatasıydı, bulgu değil.
    """
    if not need_any or (need_any & fields):
        alanlar_var = True
    else:
        alanlar_var = False
    if not alanlar_var:
        return False
    if model_tipi != "dsl_formula":
        # ML modunda birleştirmeyi model yapar: bacakların BESLENMESİ yeterli.
        # (need_any zaten yukarıda kontrol edildi; ayrıca momentum bacağı için
        #  fiyat alanı gerekir — need_any hacim bacağını temsil ediyor.)
        return True
    return need_ops <= ops


def structural_hit_rate(memory: MemoryStore, target: dict) -> "tuple[int, int]":
    """Kaç BİRİNCİL hipotez GERÇEK alpha'nın yapısını deniyor? -> (isabet, toplam).

    Araştırma verimliliğinin DÜRÜST ölçüsü: 'en iyi Sharpe' farklı bütçelerin
    maksimumunu kıyaslar ve yanıltır (bkz. compare.yaml target_structure notu).
    Bu metrik "arayıcı doğru aileyi buluyor mu?" sorusunu doğrudan ölçer.
    Yalnızca ground-truth'u bilinen kontrollü benchmark'ta anlamlıdır.

    ★ PARAMETRE VARYANTLARI SAYILMAZ (Doküman 10.1 — çoklu-testteki kuralın aynısı):
    optimizer, kabul edilen bir hipotezin pencerelerini arar ve KORELASYONLU
    kopyalar üretir; bunlar ebeveynin yapısını miras alır. Sayıya katılırsa oran
    tamamen bozulur — ölçüldü: deepseek'in 45 'backtest'inin 36'sı varyanttı ve
    isabet oranını %26 -> %41'e ŞİŞİRİYORDU; random'da hiç varyant yok (kabul
    alamayınca optimizer tetiklenmiyor) → taban tabana farklı popülasyonlar
    kıyaslanıyordu. Yalnızca birincil hipotezler sayılır.

    ★ İSABET TESTİ MODEL TİPİNE GÖRE DEĞİŞİR — bkz. _yapisal_isabet().
    Formül modunda `multiply` şart; ML modunda bacakları beslemek yeterli
    (etkileşimi model kurar). Bu ayrım olmadan ölçüt, ML kullanan yarışmacıyı
    haksız yere sıfırlıyordu.
    """
    need_ops = set(target.get("required_ops", []))
    need_any = set(target.get("required_any_field", []))
    hits = total = 0
    for (hj,) in _primary_specs(memory):
        try:
            ops, fields, model_tipi = _spec_ops_fields(hj)
        except Exception:  # noqa: BLE001 — bozuk kayıt sayımı bozmasın
            continue
        total += 1
        if _yapisal_isabet(ops, fields, model_tipi, need_ops, need_any):
            hits += 1
    return hits, total


def _primary_specs(memory: MemoryStore):
    """Birincil (parametre-varyantı OLMAYAN) backtest'lenmiş hipotezler, SIRAYLA."""
    return memory.conn.execute(
        "SELECT hypothesis_json FROM experiment "
        "WHERE sharpe IS NOT NULL AND hypothesis_json IS NOT NULL "
        "AND (relation_type IS NULL OR relation_type != 'parameter_variant') "
        "ORDER BY id")


def time_to_first_hit(memory: MemoryStore, target: dict) -> "int | None":
    """KEŞİF HIZI: kaçıncı birincil hipotezde ilk kez doğru yapı denendi? (None=hiç)

    'İsabet oranı'nı tamamlayan metrik. Oran, SÖMÜRÜ ile şişebilir: GP fitness'a
    göre evrimleşir; bir kez doğru aileyi bulunca popülasyon o yapıya yakınsar ve
    oranı yükselir — bu keşif değil, yakınsamadır. 'İlk isabete kaç deneme'
    ise saf KEŞİF hızını ölçer ve yakınsamadan etkilenmez.
    """
    need_ops = set(target.get("required_ops", []))
    need_any = set(target.get("required_any_field", []))
    for i, (hj,) in enumerate(_primary_specs(memory), start=1):
        try:
            ops, fields, model_tipi = _spec_ops_fields(hj)
        except Exception:  # noqa: BLE001
            continue
        # AYNI isabet testi (bkz. _yapisal_isabet). İki yerde iki kural olursa
        # 'isabet oranı' ile 'ilk isabet' birbiriyle çelişir.
        if _yapisal_isabet(ops, fields, model_tipi, need_ops, need_any):
            return i
    return None


@dataclass
class ContestantResult:
    label: str
    total_records: int          # üretilen her şey (duplicate/hata dahil)
    accepts: int
    duplicates: int
    compile_errors: int
    backtested: int             # ham backtest sayısı (optimizer dahil)
    distinct: int               # tekil strateji (özdeş getiriler tekilleştirilmiş)
    best_accept_sharpe: float | None
    best_dsr: float | None
    fdr_survivors: int
    tokens: int
    llm_calls_per_accept: str   # okunabilir özet
    # İYİLEŞME EĞRİSİ (Doküman 26): üretim sırasında Sharpe'lar. Kümülatif max'ı
    # 'best-so-far' eğrisidir — "aynı bütçede kim daha iyisini buluyor?"
    curve: list[float] = field(default_factory=list)
    # YAPISAL İSABET: hipotezlerin kaçı GERÇEK alpha yapısını deniyor (kontrollü
    # benchmark'ta gerçek verimlilik ölçüsü; best-Sharpe yanıltır).
    hits: int = 0
    hit_total: int = 0
    # KEŞİF HIZI: ilk isabete kaçıncı birincil hipotezde ulaştı (None = hiç)
    first_hit: "int | None" = None

    @property
    def hit_rate(self) -> "float | None":
        return (self.hits / self.hit_total) if self.hit_total else None


def _metrics(label: str, memory: MemoryStore, provider,
             target: "dict | None" = None,
             bars_per_year: int = 252) -> ContestantResult:
    stages = memory.stage_counts()
    decisions = memory.summary_by_decision()
    backtested = memory.backtested_experiments()
    rows = build_report(backtested, bars_per_year=bars_per_year)
    accepts = decisions.get("accept", 0)
    lb = memory.leaderboard(limit=1)
    tokens = (getattr(provider, "total_prompt_tokens", 0)
              + getattr(provider, "total_completion_tokens", 0))
    hits, hit_total = structural_hit_rate(memory, target) if target else (0, 0)
    first = time_to_first_hit(memory, target) if target else None
    return ContestantResult(
        label=label,
        total_records=memory.total_experiments(),
        accepts=accepts,
        duplicates=decisions.get("duplicate", 0),
        compile_errors=stages.get("compile_error", 0),
        backtested=len(backtested),
        distinct=len(rows),
        best_accept_sharpe=(lb[0][2] if lb else None),
        best_dsr=(max((r.dsr for r in rows), default=None) if rows else None),
        fdr_survivors=sum(1 for r in rows if r.survives_fdr),
        tokens=tokens,
        llm_calls_per_accept=(f"{tokens/accepts:,.0f} token/kabul" if accepts and tokens
                              else ("-" if not tokens else "kabul yok")),
        curve=memory.sharpes_in_order(),
        hits=hits,
        hit_total=hit_total,
        first_hit=first,
    )


def print_table(results: list[ContestantResult], budget: int) -> None:
    print(f"\n=== MODEL KARŞILAŞTIRMASI (deney bütçesi: {budget}/yarışmacı) ===")
    hdr = (f"{'yarışmacı':22s} {'kayıt':>6s} {'kabul':>6s} {'tekrar':>7s} "
           f"{'derl.hata':>9s} {'backtest':>9s} {'tekil':>6s} "
           f"{'en iyi Sharpe':>13s} {'en iyi DSR':>10s} {'FDR':>4s} {'token':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sh = f"{r.best_accept_sharpe:.2f}" if r.best_accept_sharpe is not None else "-"
        dsr = f"{r.best_dsr:.2f}" if r.best_dsr is not None else "-"
        print(f"{r.label:22s} {r.total_records:6d} {r.accepts:6d} {r.duplicates:7d} "
              f"{r.compile_errors:9d} {r.backtested:9d} {r.distinct:6d} "
              f"{sh:>13s} {dsr:>10s} {r.fdr_survivors:4d} {r.tokens:9,d}")
    print("\nOkuma rehberi: 'kabul' tek başına başarı DEĞİL — DSR ve FDR'a bak "
          "(çoklu-test sonrası anlamlılık). 'tekrar' yüksekse model çeşitlilik "
          "üretemiyor; 'derl.hata' yüksekse şemaya uyamıyor. Modelleri YAPISAL "
          "İSABET ve İYİLEŞME EĞRİSİ ile kıyasla ('en iyi Sharpe' yanıltıcıdır).")


def print_hit_rates(results: list[ContestantResult]) -> None:
    """YAPISAL İSABET — kontrollü benchmark'ın ASIL verimlilik metriği."""
    if not any(r.hit_total for r in results):
        return
    print("\n=== YAPISAL İSABET ORANI (arayıcı doğru aileyi buluyor mu?) ===")
    print(f"{'yarışmacı':22s} {'isabet':>8s} {'deneme':>7s} {'oran':>7s}")
    print("-" * 48)
    for r in sorted(results, key=lambda x: -(x.hit_rate or -1)):
        if not r.hit_total:
            continue
        print(f"{r.label:22s} {r.hits:8d} {r.hit_total:7d} {r.hit_rate:6.0%}")
    print("\nNeden bu metrik: 'en iyi Sharpe' farklı sayıda denemenin MAKSİMUMUNU "
          "kıyaslar — az/gürültülü deneme yapan model şansla öne geçebilir. İsabet "
          "oranı ise 'model doğru bölgeyi SİSTEMATİK buluyor mu' sorusunu ölçer "
          "(modeller arası adil karşılaştırma).")


def print_curves(results: list[ContestantResult]) -> None:
    """İyileşme eğrisi özeti: aynı bütçede kim daha hızlı/yükseğe tırmandı?"""
    print("\n=== İYİLEŞME EĞRİSİ (best-so-far Sharpe) ===")
    print(f"{'yarışmacı':22s} {'backtest':>8s} {'ilk':>7s} {'orta':>7s} {'son':>7s}")
    print("-" * 56)
    for r in results:
        if not r.curve:
            print(f"{r.label:22s} {'0':>8s} {'-':>7s} {'-':>7s} {'-':>7s}")
            continue
        b = best_so_far(r.curve)
        mid = b[len(b) // 2]
        print(f"{r.label:22s} {len(b):8d} {b[0]:+7.2f} {mid:+7.2f} {b[-1]:+7.2f}")
    print("\nOkuma: 'son' = aynı bütçede bulunan EN İYİ strateji. Yükselen eğri = "
          "model arama öğreniyor. Eğrisi en yüksek/en dik olan model, aynı bütçeyi "
          "en verimli kullanan modeldir.")


class _Kilit:
    """Aynı anda İKİ karşılaştırma koşusunu engelle.

    Neden gerekli (canlı yaşandı): yarışmacı hafıza dosyaları etiket+seed'den
    türetilir (runs/compare_<label>_s<seed>.sqlite) ve `run_contestant` her
    yarışmacıda dosyayı SİLİP yeniden kurar. İki koşu üst üste binerse biri
    diğerinin veritabanını ORTASINDA siler; sonuç çökme değil, KARIŞMIŞ
    ÖLÇÜMdür — tablo normal görünür ama sayılar iki koşunun karışımıdır.
    Bu, projede en çok uğraşılan hata sınıfının (sessizce yanlış sonuç) tam
    örneği; bu yüzden kilit çökerterek değil, açık mesajla reddediyor.
    """

    def __init__(self, yol: str) -> None:
        self._yol = yol
        self._fh = None

    def __enter__(self):
        import errno
        try:
            # O_EXCL: dosya varsa YARATMAZ, hata verir (atomik).
            fd = os.open(self._yol, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            raise SystemExit(
                f"Başka bir karşılaştırma koşusu sürüyor gibi görünüyor.\n"
                f"  Kilit: {self._yol}\n"
                f"İki koşu aynı hafıza dosyalarını paylaşır ve birbirinin "
                f"ölçümünü bozar (çökme olmaz — sayılar karışır).\n"
                f"Gerçekten koşan bir süreç yoksa kilidi sil ve tekrar dene.")
        self._fh = fd
        os.write(fd, f"pid={os.getpid()}\n".encode())
        return self

    def __exit__(self, *_exc):
        if self._fh is not None:
            os.close(self._fh)
        try:
            os.remove(self._yol)
        except OSError:
            pass
        return False


def run_contestant(contestant: dict, data, cfg, critic, db_path: str,
                   target: "dict | None" = None) -> ContestantResult:
    """Tek yarışmacıyı taze hafızayla koştur, metrikleri topla."""
    label = contestant.get("label") or contestant.get("model") or contestant["provider"]
    if os.path.exists(db_path):
        os.remove(db_path)
    provider = make_provider(contestant)
    memory = MemoryStore(db_path)
    print(f"\n########  Yarışmacı: {label}  ########")
    run_campaign(provider, data, memory, cfg, critic=critic, literature=[])
    res = _metrics(label, memory, provider, target=target,
                   bars_per_year=getattr(data, "bars_per_year", 252))
    memory.close()
    return res


def print_aggregate(runs: "dict[str, list[ContestantResult]]", budget: int,
                    seeds: list) -> None:
    """ÇOKLU-SEED özet — tek koşu yanıltır; ortalama ± aralık raporlanır."""
    print(f"\n{'='*84}\n=== ÇOKLU-SEED ÖZET ({len(seeds)} seed × {budget} deney/yarışmacı) ===")
    print(f"{'yarışmacı':22s} {'yapısal isabet':>18s} {'ilk isabet':>12s} "
          f"{'best-so-far':>18s} {'token':>9s}")
    print("-" * 84)
    rows = []
    for label, rs in runs.items():
        hits = [r.hit_rate for r in rs if r.hit_rate is not None]
        firsts = [r.first_hit for r in rs if r.first_hit is not None]
        misses = sum(1 for r in rs if r.first_hit is None)
        bests = [best_so_far(r.curve)[-1] for r in rs if r.curve]
        rows.append((mean(hits) if hits else -1.0, label, hits, firsts, misses, bests,
                     sum(r.tokens for r in rs)))
    for _, label, hits, firsts, misses, bests, toks in sorted(rows, reverse=True):
        h = (f"{mean(hits):.0%} [{min(hits):.0%}–{max(hits):.0%}]" if hits else "-")
        fh = (f"{mean(firsts):.1f}." + (f" ({misses}✗)" if misses else "")) if firsts \
            else f"hiç ({misses}✗)"
        b = (f"{mean(bests):+.2f} [{min(bests):+.2f},{max(bests):+.2f}]" if bests else "-")
        print(f"{label:22s} {h:>18s} {fh:>12s} {b:>18s} {toks:9,d}")
    print("\nOkuma:")
    print("  • 'yapısal isabet' = birincil hipotezlerin kaçı GERÇEK alpha ailesini "
          "deniyor (parametre varyantları SAYILMAZ — korelasyonlu kopyalar).")
    print("  • 'ilk isabet' = kaçıncı hipotezde doğru yapı ilk kez denendi = KEŞİF "
          "HIZI. Küçük daha iyi. (✗ = o seed'de hiç bulamadı.) Bu metrik, GP'nin "
          "yakınsama/sömürü avantajından etkilenmez.")
    print("  • 'best-so-far' = en iyi Sharpe — YANILTICIDIR (farklı deneme "
          "sayılarının maksimumu); tek başına okuma.")
    print("  • [köşeli parantez] = seed'ler arası aralık (varyans).")


# Değerlendirme ortamı compare.yaml ile değiştirildiğinde LLM'e verilen NÖTR
# evren tarifi. Belirli bir varlık sınıfı/veri alanı ima ETMEZ — yarışmacılar
# yalnızca gerçekten ellerinde olan alanlarla fikir üretsin diye.
_NOTR_EVREN = (
    "Günlük barlardan oluşan geniş kesitsel bir enstrüman evreni "
    "(fiyat: açılış/yüksek/düşük/kapanış ve işlem hacmi). Hangi piyasa ve "
    "hangi tarih aralığı olduğu BİLİNÇLİ olarak verilmiyor — genel geçer, "
    "mekanizma temelli hipotezler üret. Yalnızca sana AÇIKÇA izin verilen veri "
    "alanlarını kullan; listede olmayan bir alana dayanan fikirler elenir.")


def _filtrele(contestants: list, sadece: "str | None",
              bedava: bool) -> list:
    """Yarışmacı listesini süz. Neden gerekli:

    Liste iki AYRI soruyu cevaplayan iki grup içerir (bkz. compare.yaml):
    LLM'siz baseline'lar "LLM gerçekten arıyor mu?" (BEDAVA), LLM'ler ise
    "hangi model daha iyi?" (~$2/koşu, 3 ücretli model). Tümünü koşmak
    zorunda kalmak, bilimsel kontrolü ölçmek isteyeni para harcamaya
    mecbur ediyordu — bu yüzden çoğu koşuda baseline'lar hiç koşulmuyordu.
    """
    if sadece:
        istenen = [s.strip() for s in sadece.split(",") if s.strip()]
        secili = [c for c in contestants if c.get("label") in istenen]
        bulunamayan = set(istenen) - {c.get("label") for c in secili}
        if bulunamayan:
            mevcut = ", ".join(c.get("label", "?") for c in contestants)
            raise SystemExit(f"Bilinmeyen yarışmacı: {', '.join(sorted(bulunamayan))}\n"
                             f"Mevcut: {mevcut}")
        return secili
    if bedava:
        # `cost` YAZILMAMIŞSA ücretli varsayılır: eksik etiket yüzünden
        # habersiz para harcamak, fazladan bir yarışmacıyı atlamaktan kötüdür.
        return [c for c in contestants if c.get("cost") == "free"]
    return list(contestants)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Hipotez üreticilerini aynı veri/bütçeyle yarıştır.")
    ap.add_argument("--bedava", action="store_true",
                    help="Yalnız BEDAVA yarışmacılar (LLM'siz baseline'lar + "
                         "ücretsiz modeller). 'LLM rastgeleden iyi mi?' "
                         "sorusunu para harcamadan ölçer.")
    ap.add_argument("--sadece", metavar="ETIKET[,ETIKET]",
                    help="Yalnız bu etiketli yarışmacılar koşsun.")
    ap.add_argument("--seeds", metavar="N[,N]",
                    help="compare.yaml'daki seed listesini EZ (hızlı deneme için).")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    load_dotenv(os.path.join(HERE, ".env"))
    campaign = load_yaml("campaign.yaml")["campaign"]
    data_cfg = load_yaml("data.yaml")["data"]
    comp = load_yaml("compare.yaml")["compare"]
    cfg = build_config(campaign)

    comp = {**comp, "contestants": _filtrele(comp["contestants"], args.sadece,
                                             args.bedava)}
    if not comp["contestants"]:
        raise SystemExit("Filtre hiçbir yarışmacı bırakmadı — koşacak bir şey yok.")
    if args.seeds:
        comp = {**comp, "seeds": [int(s) for s in args.seeds.split(",")]}
    _paid = [c["label"] for c in comp["contestants"] if c.get("cost") != "free"]
    print(f"[compare] {len(comp['contestants'])} yarışmacı: "
          + ", ".join(c.get("label", "?") for c in comp["contestants"]))
    if _paid:
        # PARA HARCANACAĞI ÖNCEDEN SÖYLENİR. Sessizce token yakmak, bütçesi
        # olan biri için geri alınamaz bir sürprizdir.
        print(f"[compare] ÜCRETLİ model(ler) var: {', '.join(_paid)} "
              f"— API kredisi harcanacak. Yalnız bedavalar: --bedava")
    else:
        print("[compare] hepsi BEDAVA (API kredisi harcanmaz).")

    # BÜTÇE: karşılaştırma çok koşu yapar (yarışmacı × seed) → ayrı, küçük bütçe.
    if comp.get("budget_override"):
        cfg.max_experiments = int(comp["budget_override"])
        print(f"[compare] bütçe: {cfg.max_experiments} deney/yarışmacı (compare.yaml ezdi)")

    # Karşılaştırma kendi değerlendirme ortamını seçebilir (compare.yaml -> data).
    # Araştırma verimliliği ancak SİNYAL-VAR ortamda ölçülebilir (bkz. compare.yaml).
    if comp.get("data"):
        eski_kaynak = data_cfg.get("source")
        data_cfg = {**data_cfg, **comp["data"]}
        print(f"[compare] değerlendirme ortamı: {data_cfg['source']} (compare.yaml ezdi)")

        # ANLATIYI DA HİZALA. Veri değişip ANLATI kampanyada kalırsa, LLM'e
        # "kripto perpetual, funding_rate'e ÖNCELİK ver" denip elindeki alan
        # listesinde funding_rate olmuyor: üretilen her hipotez `disallowed_field`
        # ile eleniyor ve yarışmacı bütçesini boşa yakıyor. Gerçek koşuda
        # görüldü (nemotron 2/2 hipotezini böyle kaybetti) — çökme değil, sessiz
        # sabotaj olduğu için tabloda "kötü model" gibi görünüyordu.
        if data_cfg.get("source") != eski_kaynak:
            campaign = {**campaign, **(comp.get("campaign_override") or {})}
            if not (comp.get("campaign_override") or {}):
                campaign = {**campaign,
                            "goal": "Kesitsel günlük long-short alpha ara",
                            "literature_domain": "equity",
                            "anonymous_description": _NOTR_EVREN,
                            "universe_description": _NOTR_EVREN}
            cfg = build_config(campaign)
            if comp.get("budget_override"):
                cfg.max_experiments = int(comp["budget_override"])
            print("[compare] kampanya ANLATISI da nötrleştirildi (veri değişti): "
                  "evren tarifi/hedef/literatür artık değerlendirme ortamıyla "
                  "uyumlu. compare.yaml -> campaign_override ile özelleştirilebilir.")

    out_dir = os.path.join(HERE, comp.get("output_dir", "runs"))
    os.makedirs(out_dir, exist_ok=True)
    # Tek koşu kuralı: yarışmacı hafızaları etiket+seed'den türer, iki koşu
    # birbirinin ölçümünü sessizce bozar (bkz. _Kilit).
    with _Kilit(os.path.join(out_dir, ".compare.lock")):
        _kosu(comp, campaign, data_cfg, cfg, out_dir)


def _kosu(comp, campaign, data_cfg, cfg, out_dir) -> None:
    """Karşılaştırmanın gövdesi (kilit ALTINDA çağrılır)."""
    # Critic: adalet için varsayılan dummy; istenirse models.yaml'daki kullanılır.
    if comp.get("critic") == "models_yaml":
        from llm import make_critic
        critic = make_critic(load_yaml("models.yaml")["models"].get("quant_critic", {}))
    else:
        from agents.quant_critic import DummyCritic
        critic = DummyCritic()

    target = comp.get("target_structure")
    seeds = comp.get("seeds") or [data_cfg.get("synthetic", {}).get("seed", 7)]
    runs: "dict[str, list[ContestantResult]]" = {}
    first_curves: "dict[str, list[float]]" = {}

    for seed in seeds:
        # Aynı seed'de TÜM yarışmacılar AYNI veriyi görür (adalet).
        dcfg = data_cfg
        if "synthetic" in dcfg:
            dcfg = {**data_cfg, "synthetic": {**data_cfg["synthetic"], "seed": seed}}
        print(f"\n{'#'*74}\n#####  SEED {seed}\n{'#'*74}")
        data, _holdout = load_data(campaign, dcfg, cfg.research_fraction)

        for contestant in comp["contestants"]:
            label = contestant.get("label", "?")
            db = os.path.join(out_dir, f"compare_{label}_s{seed}.sqlite")
            # ÜRETİCİ SEED'İ DE DEĞİŞMELİ. Sabit kalırsa baseline'lar her veri
            # seed'inde AYNI hipotez dizisini üretir → arama varyansı SIFIR görünür
            # (ölçüldü: random tam %20 [20-20], hep aynı 5 hipotez). O zaman
            # 'seed tekrarı' yalnızca veriyi değiştirir, ARAMAYI değil → yanıltıcı.
            if "seed" in contestant:
                contestant = {**contestant, "seed": int(seed)}
            try:
                res = run_contestant(contestant, data, cfg, critic, db, target=target)
                runs.setdefault(label, []).append(res)
                if seed == seeds[0] and res.curve:
                    first_curves[label] = res.curve
            except Exception as e:  # noqa: BLE001 — biri çökerse diğerleri koşsun
                print(f"[{label}] KOŞU HATASI: {type(e).__name__}: {str(e)[:200]}")

    if not runs:
        return
    print_aggregate(runs, cfg.max_experiments, seeds)

    # İYİLEŞME EĞRİSİ GRAFİĞİ (makale figürü) — ilk seed, temsili.
    if first_curves:
        svg_path = os.path.join(out_dir, "improvement.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(improvement_svg(
                first_curves,
                title=f"Araştırma verimliliği — best-so-far "
                      f"(seed {seeds[0]}, {cfg.max_experiments} deney/yarışmacı)"))
        print(f"\nİyileşme eğrisi grafiği: {svg_path}")

    # Makale/rapor için markdown (çoklu-seed özeti)
    md_path = os.path.join(out_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Araştırma verimliliği ({len(seeds)} seed × "
                f"{cfg.max_experiments} deney)\n\n")
        f.write("| yarışmacı | yapısal isabet (ort [aralık]) | best-so-far "
                "(ort [aralık]) | token |\n|---|---|---|---|\n")
        for label, rs in runs.items():
            hits = [r.hit_rate for r in rs if r.hit_rate is not None]
            bests = [best_so_far(r.curve)[-1] for r in rs if r.curve]
            h = (f"{mean(hits):.0%} [{min(hits):.0%}–{max(hits):.0%}]" if hits else "-")
            b = (f"{mean(bests):+.2f} [{min(bests):+.2f}, {max(bests):+.2f}]"
                 if bests else "-")
            f.write(f"| {label} | {h} | {b} | {sum(r.tokens for r in rs):,} |\n")
    print(f"Markdown tablo: {md_path}")


if __name__ == "__main__":
    main()
