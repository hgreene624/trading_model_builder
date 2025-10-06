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


def test_sbx_handles_negative_bounds_without_complex(monkeypatch):
    """Ensure SBX crossover never produces complex numbers when bounds are negative."""

    cfg = evolutionary.EAConfig(
        crossover_op="sbx",
        crossover_rate=1.0,
        mutation_rate=0.0,
        pop_size=4,
    )

    # Freeze randomness for deterministic behaviour
    monkeypatch.setattr(evolutionary.random, "random", lambda: 0.25)

    parent_a = {"x": -10.0}
    parent_b = {"x": -6.0}
    bounds = {"x": (-12.0, -4.0)}

    child = evolutionary._crossover_configured(parent_a, parent_b, bounds, cfg)

    value = child["x"]
    assert not isinstance(value, complex)
    assert bounds["x"][0] <= value <= bounds["x"][1]


def test_mu_plus_lambda_keeps_children(monkeypatch):
    """mu+lambda replacement should retain mutated offspring for the next generation."""

    def _fake_train_general_model(*_args, **_kwargs):
        return {
            "aggregate": {
                "metrics": {
                    "trades": 10,
                    "avg_holding_days": 5.0,
                    "cagr": 0.1,
                    "calmar": 0.1,
                    "sharpe": 0.1,
                    "total_return": 0.1,
                }
            }
        }

    monkeypatch.setattr(evolutionary, "train_general_model", _fake_train_general_model)

    per_gen_params = {}

    def _progress(event, payload):
        if event == "individual_evaluated":
            per_gen_params.setdefault(payload["gen"], set()).add(
                json.dumps(payload["params"], sort_keys=True)
            )

    cfg = {
        "pop_size": 12,
        "generations": 3,
        "selection_method": "tournament",
        "tournament_k": 3,
        "replacement": "mu+lambda",
        "elitism_fraction": 0.1,
        "crossover_rate": 0.9,
        "crossover_op": "blend",
        "mutation_rate": 1.0,
        "mutation_scale": 0.4,
        "mutation_scheme": "gaussian",
        "anneal_mutation": False,
        "fitness_patience": 0,
        "shuffle_eval": False,
        "seed": 42,
    }

    evolutionary.evolutionary_search(
        strategy_dotted="tests.fake",
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-12-31",
        starting_equity=10000.0,
        param_space={"atr_n": (5, 15), "risk": (0.1, 0.9)},
        min_trades=0,
        n_jobs=1,
        progress_cb=_progress,
        config=cfg,
    )

    assert len(per_gen_params) >= 2, "Expected multiple generations to run"
    first_gen = per_gen_params.get(0, set())
    assert first_gen, "Generation 0 should have evaluated individuals"
    later_diversity = set().union(*(per_gen_params[g] for g in per_gen_params if g >= 1))
    assert later_diversity - first_gen, "Later generations should include new parameter combinations"
