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


def test_scoring_prefers_test_metrics(monkeypatch, tmp_path):
    """Scoring should favor holdout metrics when a test range is provided."""

    call_log: list[tuple[str, str, int]] = []

    def _fake_train_general_model(_strategy, _tickers, start, end, _equity, params):
        # Track order and window so we can confirm both train/test windows are used.
        call_log.append((str(start), str(end), params.get("x", 0)))
        base = {
            "trades": 8,
            "avg_holding_days": 4.0,
        }
        x_val = float(params.get("x", 0))
        if str(start).startswith("2020-04"):
            score = 0.15 + 0.35 * x_val  # holdout rewards higher x
        else:
            score = 0.60 - 0.30 * x_val  # training prefers lower x
        base.update(
            {
                "total_return": score,
                "cagr": score,
                "calmar": score,
                "sharpe": score,
            }
        )
        return {"aggregate": {"metrics": base}}

    pop_values = iter([
        {"x": 1},
        {"x": 2},
        {"x": 1},
        {"x": 2},
    ])

    def _fake_random_param(space):
        try:
            value = next(pop_values)
        except StopIteration:
            value = {"x": 2}
        return dict(value)

    monkeypatch.setattr(evolutionary, "train_general_model", _fake_train_general_model)
    monkeypatch.setattr(evolutionary, "random_param", _fake_random_param)

    log_file = tmp_path / "ea_holdout.jsonl"

    cfg = {
        "pop_size": 2,
        "generations": 1,
        "selection_method": "tournament",
        "tournament_k": 2,
        "replacement": "generational",
        "elitism_fraction": 0.5,
        "crossover_rate": 0.0,
        "mutation_rate": 0.0,
        "mutation_scale": 0.1,
        "anneal_mutation": False,
        "fitness_patience": 0,
        "shuffle_eval": False,
        "seed": 1,
    }

    results = evolutionary.evolutionary_search(
        strategy_dotted="tests.fake",
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-03-31",
        test_start="2020-04-01",
        test_end="2020-06-30",
        starting_equity=10000.0,
        param_space={"x": (1, 2)},
        min_trades=0,
        n_jobs=1,
        random_inject_frac=0.0,
        log_file=str(log_file),
        config=cfg,
    )

    assert results, "EA should produce scored candidates"
    _best_params, best_score = results[0]
    assert best_score > 0.0

    # Each individual should have triggered both train and test evaluations
    assert len(call_log) == 4
    assert any(start.startswith("2020-04") for start, _end, _ in call_log)

    log_lines = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    eval_payloads = [rec["payload"] for rec in log_lines if rec.get("event") == "individual_evaluated"]
    assert eval_payloads
    for payload in eval_payloads:
        assert "train_metrics" in payload
        if "test_metrics" in payload:
            assert payload["metrics"] == payload["test_metrics"], "Metrics should reflect holdout values"
        blend = payload.get("score_blend") or {}
        if blend:
            assert blend.get("holdout_component", 0.0) >= blend.get("train_bonus", 0.0)
            assert blend.get("penalty", 0.0) >= 0.0
            expected_component = blend.get("holdout_weight", 0.0) * blend.get("test_score", 0.0)
            assert blend.get("holdout_component") == pytest.approx(expected_component)


def test_holdout_blend_penalizes_overfit(monkeypatch):
    """Candidates that collapse on the holdout window should score lower than balanced ones."""

    def _fake_train_general_model(_strategy, _tickers, start, _end, _equity, params):
        base_metrics = {
            "trades": 12,
            "avg_holding_days": 5.0,
        }
        x_val = params.get("x", 0)
        if str(start).startswith("2020-04"):
            score = 0.1 if x_val == 0 else 0.4
        else:
            score = 0.8 if x_val == 0 else 0.4
        base_metrics.update(
            {
                "total_return": score,
                "cagr": score,
                "calmar": score,
                "sharpe": score,
            }
        )
        return {"aggregate": {"metrics": base_metrics}}

    seq = iter([
        {"x": 0},
        {"x": 1},
    ])

    def _fake_random_param(_space):
        try:
            return dict(next(seq))
        except StopIteration:
            return {"x": 1}

    monkeypatch.setattr(evolutionary, "train_general_model", _fake_train_general_model)
    monkeypatch.setattr(evolutionary, "random_param", _fake_random_param)

    cfg = {
        "pop_size": 2,
        "generations": 1,
        "selection_method": "tournament",
        "tournament_k": 2,
        "replacement": "generational",
        "elitism_fraction": 0.5,
        "crossover_rate": 0.0,
        "mutation_rate": 0.0,
        "mutation_scale": 0.1,
        "anneal_mutation": False,
        "fitness_patience": 0,
        "shuffle_eval": False,
        "seed": 7,
    }

    results = evolutionary.evolutionary_search(
        strategy_dotted="tests.fake",
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-03-31",
        test_start="2020-04-01",
        test_end="2020-06-30",
        starting_equity=10000.0,
        param_space={"x": (0, 1)},
        min_trades=0,
        n_jobs=1,
        random_inject_frac=0.0,
        config=cfg,
    )

    assert results, "EA should evaluate candidates"
    best_params, best_score = results[0]
    assert best_params["x"] == 1, "Balanced candidate should win when holdout collapses"
    assert best_score > 0.0


def test_holdout_blend_penalizes_train_shortfall(monkeypatch):
    """Holdout scoring should demote genomes that crater on the training window."""

    def _fake_train_general_model(_strategy, _tickers, start, _end, _equity, params):
        base_metrics = {
            "trades": 10,
            "avg_holding_days": 4.0,
        }
        x_val = params.get("x", 0)
        if str(start).startswith("2020-04"):
            # Holdout rewards higher x values
            score = 0.9 if x_val == 0 else 0.7
        else:
            # Training punishes x=0 while keeping x=1 healthy
            score = -0.8 if x_val == 0 else 0.6
        base_metrics.update(
            {
                "total_return": score,
                "cagr": score,
                "calmar": score,
                "sharpe": score,
            }
        )
        return {"aggregate": {"metrics": base_metrics}}

    seq = iter([
        {"x": 0},
        {"x": 1},
    ])

    def _fake_random_param(_space):
        try:
            return dict(next(seq))
        except StopIteration:
            return {"x": 1}

    monkeypatch.setattr(evolutionary, "train_general_model", _fake_train_general_model)
    monkeypatch.setattr(evolutionary, "random_param", _fake_random_param)

    cfg = {
        "pop_size": 2,
        "generations": 1,
        "selection_method": "tournament",
        "tournament_k": 2,
        "replacement": "generational",
        "elitism_fraction": 0.5,
        "crossover_rate": 0.0,
        "mutation_rate": 0.0,
        "mutation_scale": 0.1,
        "anneal_mutation": False,
        "fitness_patience": 0,
        "shuffle_eval": False,
        "seed": 9,
    }

    results = evolutionary.evolutionary_search(
        strategy_dotted="tests.fake",
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-03-31",
        test_start="2020-04-01",
        test_end="2020-06-30",
        starting_equity=10000.0,
        param_space={"x": (0, 1)},
        min_trades=0,
        n_jobs=1,
        random_inject_frac=0.0,
        config=cfg,
    )

    assert results, "EA should evaluate candidates"
    best_params, best_score = results[0]
    assert best_params["x"] == 1, "Strategy with healthier training window should win"
    assert best_score > 0.0
