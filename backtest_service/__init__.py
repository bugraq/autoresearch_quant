"""
Backtest Servisi — bağımsız, model-agnostik backtest servisi (hocanın ilk hedefi).

    from backtest_service import run
    rapor = run(spec, data)
    print(rapor.summary())

Model formül / istatistiksel / ML olabilir — çağrı aynıdır.
LLM ve orchestrator'dan bağımsızdır; tek başına kullanılabilir.
"""
from backtest_service.service import (
    BacktestReport,
    BacktestServiceError,
    LeakageError,
    run,
)

__all__ = ["run", "BacktestReport", "BacktestServiceError", "LeakageError"]
