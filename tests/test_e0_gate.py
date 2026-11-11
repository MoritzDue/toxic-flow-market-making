"""E0 verification gate at reduced path counts (full-precision numbers: results/e0.json).
The automated threshold is |z| < 3 so the test does not fail by chance one time in twenty;
the 95% statement in the report is made on the full run."""
import pytest
from toxicmm.params import Params
from experiments import e0_verification as e0


@pytest.mark.parametrize("name", list(e0.PARAM_SETS))
def test_e0a_e0b_monte_carlo_matches_hjb(name):
    res = e0.check_values(e0.PARAM_SETS[name], 40_000, 77)
    for mm, v in res.items():
        assert abs(v["z"]) < 3, (name, mm, v)


def test_e0a_strategic():
    res = e0.check_values(Params(), 40_000, 78, strategic=True)
    assert abs(res["Oracle"]["z"]) < 3


def test_e0c_single_fill_not_identifiable():
    res = e0.classifier(Params(), n_rec=500)
    assert res["prior_0.5"]["accuracy"] < 0.9
    assert abs(res["simulated_balanced_accuracy"] - res["prior_0.5"]["accuracy"]) < 0.02


def test_e0d_no_look_ahead():
    res = e0.look_ahead(Params(), n_rec=50)
    for mm, v in res.items():
        assert v["same_belief_before"] and v["same_fills_before"] and v["differs_after"], mm
