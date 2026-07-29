"""Veri katmanı — sentetik üreteçler + DataAdapter (sentetik/gerçek tak-çalıştır)."""
from data.adapter import (
    DataAdapter,
    SyntheticAdapter,
    YFinanceAdapter,
    bars_per_year_from_config,
    make_adapter,
)
from data.synthetic import (
    MarketData,
    concat_market,
    gen_cross_sectional_momentum,
    gen_random,
    gen_short_term_reversal,
    split_by_fraction,
)

__all__ = [
    "MarketData",
    "split_by_fraction",
    "concat_market",
    "gen_random",
    "gen_cross_sectional_momentum",
    "gen_short_term_reversal",
    "DataAdapter",
    "SyntheticAdapter",
    "YFinanceAdapter",
    "make_adapter",
    "bars_per_year_from_config",
]
