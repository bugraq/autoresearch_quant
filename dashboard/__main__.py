"""`python -m dashboard` giriş noktası — dashboard.html üretir.

(`python -m dashboard.report` de çalışır ama Python bir RuntimeWarning basar;
paket girişi olan bu dosya uyarısızdır. Menü [5] zaten hazır html'i açar.)
"""
from dashboard.report import _cli_main

_cli_main()
