import json

import pytest

from odsim import SimConfig, Site


def test_defaults_are_valid():
    SimConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"n_agents": -5},
        {"n_steps": 1.5},
        {"p_overdose": 1.2},
        {"frac_housed": -0.1},
        {"naloxone_radius": -1.0},
        {"sites": [{"x": 99, "y": 0, "kind": "treatment"}]},
    ],
)
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_bad_site_kind_rejected():
    with pytest.raises(ValueError):
        Site(1, 1, "hospital")


def test_sites_from_dicts_become_site_objects():
    cfg = SimConfig(sites=[{"x": 1, "y": 2, "kind": "ssp"}])
    assert cfg.sites == (Site(1, 2, "ssp"),)


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        SimConfig.from_dict({"n_agents": 10, "typo_param": 3})
    with pytest.raises(ValueError):
        SimConfig().with_overrides(typo_param=3)


def test_overrides_return_new_config():
    base = SimConfig()
    changed = base.with_overrides(p_enroll=0.2)
    assert changed.p_enroll == 0.2
    assert base.p_enroll != 0.2


def test_extra_sites_append():
    base = SimConfig(sites=[Site(1, 1, "treatment")])
    changed = base.with_overrides(extra_sites=[{"x": 2, "y": 2, "kind": "ssp"}])
    assert changed.sites == (Site(1, 1, "treatment"), Site(2, 2, "ssp"))


def test_json_round_trip(tmp_path):
    cfg = SimConfig(n_agents=42, sites=[Site(3, 4, "treatment")])
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert SimConfig.from_json(path) == cfg
