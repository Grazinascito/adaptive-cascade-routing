"""End-to-end smoke test of the benchmark orchestrator (PR5) against fake HTTP.

A threading http.server mimics vLLM's response shapes on /v1/systemone
(per-alias probabilities, as System1Engine expects) and
/v1/chat/completions (JSON with reasoning/category/revisor_insight, as
System2Engine expects). Canned behavior: System 1 is confident (0.97) and
correct on the head, wrong with low confidence on the tail (0.72 / 0.88);
System 2 gets everything right — or misses the tail in `s2_fails_tail`
mode, for the pilot gate's failure path.
"""

import copy
import json
import threading
from collections.abc import Iterator
from dataclasses import asdict, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from src.benchmark import run as run_module
from src.benchmark import runner
from src.benchmark.runner import BenchmarkHarness, BenchmarkSettings
from src.data.loader import BenchmarkSets, CatalogSample
from src.engine.system1 import assign_aliases
from src.metrics.evaluator import BenchmarkRecord

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "daru_sample.json"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark_config.yaml"
ALIAS_TAIL_P_TOP = {"CERV SKOL 269ML": 0.72, "CEBOLA ROXA KG": 0.88}
DECOY_CATEGORY = "MERCEARIA"
DECOY_FALLBACK = "FRIOS E LATICINIOS"
DECODE_TOKENS_PER_CALL = 128


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _decoy_for(category: str) -> str:
    return DECOY_FALLBACK if category == DECOY_CATEGORY else DECOY_CATEGORY


class _FakeDaruBackend:
    """Canned responses from both endpoints, deterministic by item text."""

    def __init__(self, categories: list[str], *, s2_fails_tail: bool = False) -> None:
        alias_to_category = assign_aliases(categories)
        self._category_to_alias = {v: k for k, v in alias_to_category.items()}
        self._aliases = list(alias_to_category)
        self._by_text: dict[str, str] = {}
        self.s2_fails_tail = s2_fails_tail
        self.systemone_calls = 0
        self.chat_calls = 0

    def register(self, text: str, category: str) -> None:
        self._by_text[text] = category

    def systemone(self, body: dict[str, object]) -> dict[str, object]:
        self.systemone_calls += 1
        text = str(body["state"])
        category = self._by_text[text]
        if len(text.strip()) <= 20:
            top_alias = self._category_to_alias[_decoy_for(category)]
            p_top = ALIAS_TAIL_P_TOP.get(text, 0.5)
        else:
            top_alias = self._category_to_alias[category]
            p_top = 0.97
        rest = (1.0 - p_top) / (len(self._aliases) - 1)
        probabilities = {alias: p_top if alias == top_alias else rest for alias in self._aliases}
        return {"answers": {"category": {"probabilities": probabilities}}}

    def chat(self, body: dict[str, object]) -> dict[str, object]:
        self.chat_calls += 1
        messages = body["messages"]
        assert isinstance(messages, list)
        user_content = str(messages[1]["content"])
        text = user_content.split("Item: ", 1)[1].split("\n", 1)[0]
        category = self._by_text[text]
        predicted = (
            _decoy_for(category) if self.s2_fails_tail and len(text.strip()) <= 20 else category
        )
        content = json.dumps(
            {
                "reasoning": f"analyzed '{text}' step by step",
                "category": predicted,
                "revisor_insight": "generic fake rule",
            },
            ensure_ascii=False,
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": DECODE_TOKENS_PER_CALL},
        }


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    backend: _FakeDaruBackend

    def do_POST(self) -> None:  # noqa: N802 - name required by http.server
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.path.endswith("/systemone"):
            payload = self.backend.systemone(body)
        elif self.path.endswith("/chat/completions"):
            payload = self.backend.chat(body)
        else:
            self.send_error(404)
            return
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def backend() -> _FakeDaruBackend:
    categories = list(_load_config()["dataset"]["categories"])
    return _FakeDaruBackend(categories)


@pytest.fixture()
def fake_server(backend: _FakeDaruBackend) -> Iterator[str]:
    _FakeVLLMHandler.backend = backend
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture()
def samples(backend: _FakeDaruBackend) -> list[CatalogSample]:
    """12 fixture items: 10 head + 2 tail, all within the 12 categories."""
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    categories = set(_load_config()["dataset"]["categories"])
    eligible = [row for row in rows if row["categoria"] in categories]
    head = [row for row in eligible if len(row["description"].strip()) > 20]
    tail = [row for row in eligible if len(row["description"].strip()) <= 20]
    chosen = head[:10] + tail[:2]
    built = [
        CatalogSample(
            sample_id=sample_id,
            raw_text=row["description"],
            canonical_category=row["categoria"],
            is_long_tail=len(row["description"].strip()) <= 20,
        )
        for sample_id, row in enumerate(chosen)
    ]
    for sample in built:
        backend.register(sample.raw_text, sample.canonical_category)
    return built


@pytest.fixture()
def settings(fake_server: str) -> BenchmarkSettings:
    return replace(BenchmarkSettings.from_config(_load_config()), base_url=fake_server)


@pytest.fixture()
def harness(settings: BenchmarkSettings) -> Iterator[BenchmarkHarness]:
    engine = BenchmarkHarness(settings)
    yield engine
    engine.close()


def test_every_record_dict_carries_all_16_fields(
    harness: BenchmarkHarness, samples: list[CatalogSample]
) -> None:
    field_names = {field.name for field in fields(BenchmarkRecord)}
    assert len(field_names) == 16

    scenarios = [
        harness.run_system1_scenario(samples),
        harness.run_system2_scenario(samples),
        harness.run_cascade_scenario(samples, alpha=0.85),
    ]
    for records in scenarios:
        assert len(records) == len(samples)
        for record in records:
            assert set(asdict(record)) == field_names


def test_system1_scenario_never_routes_and_final_matches_s1(
    harness: BenchmarkHarness, samples: list[CatalogSample]
) -> None:
    records = harness.run_system1_scenario(samples)

    assert all(not record.routed_to_s2 for record in records)
    assert all(record.s2_prediction is None for record in records)
    assert all(record.final_prediction == record.s1_prediction for record in records)
    assert sum(record.is_correct for record in records) == 10
    assert all(record.s1_p_top > 0.9 for record in records if not record.is_long_tail)


def test_system2_scenario_leaves_s1_blank_and_routes_everything(
    harness: BenchmarkHarness, samples: list[CatalogSample]
) -> None:
    records = harness.run_system2_scenario(samples)

    assert all(record.routed_to_s2 for record in records)
    assert all(record.s1_prediction == "" for record in records)
    assert all(record.s1_p_top == 0.0 and record.s1_entropy == 0.0 for record in records)
    assert all(record.final_prediction == record.s2_prediction for record in records)
    assert all(record.s2_decode_tokens == DECODE_TOKENS_PER_CALL for record in records)
    assert all(record.is_correct for record in records)


def test_cascade_routes_exactly_the_low_confidence_items(
    harness: BenchmarkHarness, samples: list[CatalogSample]
) -> None:
    records = harness.run_cascade_scenario(samples, alpha=0.85)

    for record in records:
        assert record.routed_to_s2 == (record.s1_p_top < 0.85)

    routed = [record for record in records if record.routed_to_s2]
    assert len(routed) == 1
    assert routed[0].raw_text == "CERV SKOL 269ML"
    assert routed[0].s1_p_top == pytest.approx(0.72)
    assert routed[0].final_prediction == routed[0].s2_prediction == "BEBIDAS ALCOOLICAS"
    assert routed[0].s2_decode_tokens == DECODE_TOKENS_PER_CALL

    unrouted = [record for record in records if not record.routed_to_s2]
    assert len(unrouted) == 11
    assert all(record.s2_prediction is None for record in unrouted)
    assert all(record.final_prediction == record.s1_prediction for record in unrouted)
    assert sum(record.is_correct for record in records) == 11


def test_alpha_sweep_routed_rate_is_monotone_and_best_alpha_wins(
    harness: BenchmarkHarness, samples: list[CatalogSample]
) -> None:
    validation = harness.run_validation_phase(samples)
    sweep = runner.alpha_sweep(validation)
    alphas = sweep["alphas"]

    assert set(alphas) == {"0.70", "0.80", "0.85", "0.90"}
    assert alphas["0.70"]["routed_rate"] <= alphas["0.85"]["routed_rate"]
    assert alphas["0.85"]["routed_rate"] <= alphas["0.90"]["routed_rate"]
    assert alphas["0.70"]["decode_tokens"] <= alphas["0.85"]["decode_tokens"]
    assert alphas["0.85"]["decode_tokens"] <= alphas["0.90"]["decode_tokens"]
    assert alphas["0.70"]["routed_rate"] == 0.0
    assert alphas["0.90"]["routed_rate"] == pytest.approx(2 / 12)
    assert alphas["0.90"]["accuracy"] == pytest.approx(1.0)
    assert alphas["0.70"]["accuracy"] == pytest.approx(10 / 12)
    assert sweep["best_alpha"] == pytest.approx(0.90)


def test_best_alpha_tie_breaks_on_lower_decode_tokens() -> None:
    metrics = {
        "0.70": {"accuracy": 0.9, "routed_rate": 0.1, "decode_tokens": 10},
        "0.85": {"accuracy": 0.9, "routed_rate": 0.3, "decode_tokens": 5},
        "0.90": {"accuracy": 0.8, "routed_rate": 0.5, "decode_tokens": 1},
    }

    assert runner.best_alpha(metrics) == pytest.approx(0.85)


def test_payload_shape_gates_and_written_json(
    harness: BenchmarkHarness,
    samples: list[CatalogSample],
    settings: BenchmarkSettings,
    tmp_path: Path,
) -> None:
    validation = harness.run_validation_phase(samples)
    records_a = harness.run_system1_scenario(samples)
    records_b = harness.run_system2_scenario(samples)
    records_c = harness.run_cascade_scenario(samples, alpha=0.85)

    payload = runner.assemble_payload(
        config=_load_config(),
        settings=settings,
        started_at="2026-11-15T00:00:00+00:00",
        finished_at="2026-11-15T00:01:00+00:00",
        git_rev="deadbeef",
        alpha_used=0.85,
        alpha_source="validation_sweep",
        validation_records=validation,
        records_a=records_a,
        records_b=records_b,
        records_c=records_c,
    )

    assert set(payload) == {"meta", "scenarios", "alpha_sweep", "summaries", "kpis"}
    assert set(payload["scenarios"]) == {"SC-A", "SC-B", "SC-C", "VALIDATION"}
    assert all(len(records) == 12 for records in payload["scenarios"].values())
    assert payload["meta"]["git_rev"] == "deadbeef"
    assert payload["meta"]["model"] == settings.model
    assert payload["meta"]["config"]["router"]["alpha"] == 0.85

    kpis = payload["kpis"]
    assert kpis["tail_gap_pp"]["achieved"] == pytest.approx(100.0)
    assert kpis["tail_gap_pp"]["passed"] is True
    assert kpis["recovery_share"]["achieved"] == pytest.approx(0.5)
    assert kpis["recovery_share"]["passed"] is True
    assert kpis["token_share_vs_s2"]["achieved"] == pytest.approx(128 / 1536)
    assert kpis["token_share_vs_s2"]["passed"] is True

    output = tmp_path / "raw_benchmark_data.json"
    runner.write_results(payload, output)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    assert set(reloaded) == set(payload)
    assert len(reloaded["scenarios"]["SC-C"]) == 12
    assert reloaded["alpha_sweep"]["best_alpha"] == pytest.approx(0.90)


def test_warmup_discards_five_requests_per_engine_before_each_scenario(
    harness: BenchmarkHarness, samples: list[CatalogSample], backend: _FakeDaruBackend
) -> None:
    harness.run_system1_scenario(samples)

    assert backend.systemone_calls == 5 + len(samples)
    assert backend.chat_calls == 5


def test_pilot_gate_passes_then_fails_when_s2_loses_the_tail(
    harness: BenchmarkHarness,
    samples: list[CatalogSample],
    backend: _FakeDaruBackend,
) -> None:
    pilot = runner.select_pilot_samples(samples, n_head=10, n_tail=2)
    assert len(pilot) == 12
    assert sum(sample.is_long_tail for sample in pilot) == 2

    passed = runner.run_pilot(harness, samples, required_gap_pp=10.0, n_head=10, n_tail=2)
    assert passed.passed is True
    assert passed.s1_tail_accuracy == pytest.approx(0.0)
    assert passed.s2_tail_accuracy == pytest.approx(1.0)
    assert passed.tail_gap_pp == pytest.approx(100.0)

    backend.s2_fails_tail = True
    failed = runner.run_pilot(harness, samples, required_gap_pp=10.0, n_head=10, n_tail=2)
    assert failed.passed is False
    assert failed.s2_tail_accuracy == pytest.approx(0.0)
    assert failed.tail_gap_pp == pytest.approx(0.0)
    assert "GATE: FAILED" in runner.format_pilot_report(failed)


def test_cli_pilot_exit_code_and_gate_artifact(
    fake_server: str,
    backend: _FakeDaruBackend,
    samples: list[CatalogSample],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = copy.deepcopy(_load_config())
    config["serving"]["base_url"] = fake_server
    monkeypatch.setattr(runner, "load_config", lambda path: config)
    monkeypatch.setattr(
        runner,
        "build_benchmark_sets",
        lambda settings, xlsx_path: BenchmarkSets(validation=samples, test=samples),
    )
    monkeypatch.setattr(runner, "PILOT_HEAD", 10)
    monkeypatch.setattr(runner, "PILOT_TAIL", 2)
    monkeypatch.chdir(tmp_path)

    assert run_module.main(["--config", "cfg.yaml", "--pilot"]) == 0
    assert "GATE: PASSED" in capsys.readouterr().out
    gate = json.loads((tmp_path / "results" / "pilot_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True

    backend.s2_fails_tail = True
    assert run_module.main(["--config", "cfg.yaml", "--pilot"]) == 1
    assert "GATE: FAILED" in capsys.readouterr().out
