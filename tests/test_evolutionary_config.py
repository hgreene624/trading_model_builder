import json
from pathlib import Path

import pytest

from src.optimization import evolutionary


@pytest.mark.parametrize("fitness_value", [0.25])
def test_ea_config_smoke(monkeypatch, tmp_path, fitness_value):
    """Smoke-test the new EA config path with early stopping and logging."""

    def _fake_train_general_model(*_args, **_kwargs):
        return {
            "aggregate": {
                "metrics": {
                    "trades": 12,
                    "avg_holding_days": 5.0,
                    "cagr": fitness_value,
                    "calmar": fitness_value,
                    "sharpe": fitness_value,
                    "total_return": fitness_value,
                }
            }
        }

    monkeypatch.setattr(evolutionary, "train_general_model", _fake_train_general_model)

    log_file = tmp_path / "ea_test.jsonl"

    cfg = {
        "pop_size": 16,
        "generations": 3,
        "selection_method": "tournament",
        "tournament_k": 3,
        "replacement": "mu+lambda",
        "elitism_fraction": 0.05,
        "crossover_rate": 0.9,
        "crossover_op": "blend",
        "mutation_rate": 0.15,
        "mutation_scale": 0.20,
        "mutation_scheme": "gaussian",
        "anneal_mutation": False,
        "anneal_floor": 0.05,
        "fitness_patience": 1,
        "shuffle_eval": False,
        "seed": 1,
    }

    results = evolutionary.evolutionary_search(
        strategy_dotted="tests.fake",
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
        starting_equity=10000.0,
        param_space={"atr_n": (5, 10)},
        min_trades=0,
        n_jobs=1,
        log_file=str(log_file),
        config=cfg,
    )

    assert results, "EA should return at least one candidate"

    # Parse log file for config + completion metadata
    log_lines = [json.loads(line) for line in Path(log_file).read_text().splitlines() if line.strip()]
    cfg_payloads = [rec["payload"] for rec in log_lines if rec.get("event") == "ea_config"]
    assert cfg_payloads, "ea_config event should be logged at gen 0"
    elite_count = cfg_payloads[0].get("elite_count")
    assert isinstance(elite_count, int) and elite_count >= 1

    done_payloads = [rec["payload"] for rec in log_lines if rec.get("event") == "session_end"]
    assert done_payloads, "session_end event should be logged"
    done = done_payloads[0]
    assert done.get("stopped_early"), "Early stopping should trigger with patience=1"
    assert done.get("generations_ran") <= cfg["generations"]
