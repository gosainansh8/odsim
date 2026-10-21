import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from odsim.web.app import app  # noqa: E402

client = TestClient(app)


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "odsim" in page.text


def test_parameters_listed_with_base_values():
    data = client.get("/api/parameters").json()
    names = {p["name"] for p in data["parameters"]}
    assert {"frac_housed", "naloxone_radius", "bystander_effect"} <= names
    for p in data["parameters"]:
        assert p["min"] <= p["base"] <= p["max"]


def test_simulate_returns_comparison():
    body = {"intervention": {"frac_housed": 0.9}, "n_agents": 300, "reps": 2}
    data = client.post("/api/simulate", json=body).json()
    assert set(data["summary"]) == {"baseline", "intervention"}
    assert data["summary"]["intervention"]["hazard_ratio"] > 0
    assert data["curves"]["baseline"]["time"][0] == 0
    assert len(data["map"]["agents"]) == 300
    assert data["map"]["sites"]


def test_identical_requests_are_cached():
    body = {"intervention": {"p_enroll": 0.01}, "n_agents": 200, "reps": 1, "seed": 9}
    assert client.post("/api/simulate", json=body).json() == client.post("/api/simulate", json=body).json()


@pytest.mark.parametrize(
    "body",
    [
        {"intervention": {"p_fatal": 0.5}},  # not adjustable
        {"intervention": {"frac_housed": 2.0}},  # out of range
        {"n_agents": 10_000_000},  # too expensive
        {"reps": 1000},
    ],
)
def test_rejects_bad_requests(body):
    assert client.post("/api/simulate", json=body).status_code == 422
