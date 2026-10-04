import numpy as np

from bayes_football.basic_model import score_log_rates


def test_score_log_rates_follow_sign_convention() -> None:
    attack = np.array([0.2, -0.2])
    defence = np.array([-0.1, 0.1])
    home_log, away_log = score_log_rates(
        np.array([0]), np.array([1]), 0.3, attack, defence
    )
    np.testing.assert_allclose(home_log, [0.6])
    np.testing.assert_allclose(away_log, [-0.3])


def test_sum_to_zero_example_is_identified() -> None:
    raw = np.array([-0.5, 0.2, 0.8])
    centred = raw - raw.mean()
    assert abs(centred.sum()) < 1e-12
