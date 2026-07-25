"""
Model kaydı — DSL özelliklerini (X) SÜREKLI bir tahmin skoruna (sinyal) çeviren
öğrenen modeller. Hoca yönü: 'hipotez → MODEL → backtest', model basit cebir
(lineer regresyon) ya da olasılıksal (naive bayes) olabilir; sonra ML/LSTM.

Sözleşme: fit_predict(model_type, X_tr, y_tr, X_pred, params) -> np.ndarray
  X_tr/X_pred: (n_örnek, n_özellik) sayısal matris
  y_tr: (n_örnek,) ileriki getiri (regresyon hedefi)
  dönüş: (n_pred,) SÜREKLI skor — büyük = daha çok long. Sıralanıp portföye girer.

SIZINTI: model yalnızca burada verilen (geçmiş) örneklerle fit edilir; walk-forward
çağıran taraf (backtest/model_signal.py) train/pred ayrımını embargo ile yapar.
Bu fonksiyon geleceği GÖRMEZ — sadece kendine verilen X_tr/y_tr'yi kullanır.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB

# dsl_formula burada YOK: o, model kutusunu atlar (sinyal doğrudan DSL ifadesidir).
# Hoca tarifi: model "formül de olabilir, istatistiksel veya ML modeli de olabilir".
#   formül        -> dsl_formula (models/ dışında; sinyal DSL ifadesidir)
#   istatistiksel -> linear_regression, ridge, naive_bayes
#   ML            -> random_forest, gradient_boosting  (ileride LSTM: Qlib/torch)
SUPPORTED_MODELS = {"linear_regression", "ridge", "naive_bayes",
                    "random_forest", "gradient_boosting"}

# TEKRAR ÜRETİLEBİLİRLİK: model nesnesi değil, ADI + params saklanır (JSON'a yazılır).
# Rastgelelik içeren modellere sabit seed verilir; aynı spec -> aynı sonuç.
_SEED = 42


def fit_predict(model_type: str, X_tr: np.ndarray, y_tr: np.ndarray,
                X_pred: np.ndarray, params: "dict | None" = None) -> np.ndarray:
    params = params or {}
    if model_type == "linear_regression":
        # Tahmini getiri (sürekli). Skor = beklenen getiri; sıralamayla long/short.
        m = LinearRegression()
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_pred), dtype=float)

    if model_type == "ridge":
        # Düzenlileştirilmiş lineer: çok/korelasyonlu özellikte overfit'i frenler.
        m = Ridge(alpha=float(params.get("alpha", 1.0)))
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_pred), dtype=float)

    if model_type == "random_forest":
        # ML: doğrusal-olmayan etkileşimleri yakalar. Derinlik SINIRLI tutulur —
        # finansal veride sinyal/gürültü çok düşük, derin ağaç ezberler.
        m = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 4)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            random_state=_SEED, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_pred), dtype=float)

    if model_type == "gradient_boosting":
        m = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_state=_SEED)
        m.fit(X_tr, y_tr)
        return np.asarray(m.predict(X_pred), dtype=float)

    if model_type == "naive_bayes":
        # Sınıf = ileriki getirinin İŞARETİ (yukarı/aşağı). Skor = P(yukarı) - 0.5
        # → yine SÜREKLI sayı (olasılık), sıralanabilir.
        cls = (y_tr > 0).astype(int)
        if len(np.unique(cls)) < 2:
            # Tek sınıf (hepsi + ya da hepsi -): ayrım yok, nötr skor.
            return np.zeros(len(X_pred), dtype=float)
        m = GaussianNB()
        m.fit(X_tr, cls)
        proba = m.predict_proba(X_pred)
        up_col = list(m.classes_).index(1)
        return np.asarray(proba[:, up_col] - 0.5, dtype=float)

    raise ValueError(f"Bilinmeyen model tipi: {model_type!r} "
                     f"(desteklenen: {sorted(SUPPORTED_MODELS)})")
