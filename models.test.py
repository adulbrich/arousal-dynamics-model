import pytest
import numpy as np
from models import (
    wake_effort,
    total_sleep_drive,
    nonphotic_drive,
    photoreceptor_conversion_rate,
    photic_drive,
    circadian_phase,
    # mean_population_firing_rate,
    # state,
    # sigmoid,
    # circadian_drive,
    # model
)

# TODO: update for latest revision of models.py

@pytest.fixture
def model_state():
    """Initial state vector for model testing."""
    return [ -4.55, -0.07, 13.29, -0.14, -1.07, 0.10, -5.00e-06]  # V_v, V_m, H, X, Y, P, Theta_L

def test_wake_effort():
    """Test wake effort calculation function."""
    # Test normal case (no forced wake)
    assert wake_effort(1.0, 0) == 0.0
    
    # Test forced wake case
    assert wake_effort(30, 1) == 52.63

def test_total_sleep_drive():
    """Test total sleep drive calculation function."""
    # Test normal cases
    assert total_sleep_drive(1.0, 1.0) == -9.8
    assert total_sleep_drive(25.0, 2.5) == 13.45
    
    # Test edge case with zero inputs
    assert total_sleep_drive(0, 0) == -10.3

def test_nonphotic_drive():
    """Test nonphotic drive calculation function."""
    # Test normal case
    assert nonphotic_drive(-0.14, 1) == np.float64(0.6284505494007543)
    assert nonphotic_drive(0.14, 1) == np.float64(0.038216117265912494)
    assert nonphotic_drive(0.5, 0) == np.float64(-6.053049160326118e-05)
    
    # Test edge case with zero inputs
    assert nonphotic_drive(0, 1) == np.float64(0.33333333333333337)
    assert nonphotic_drive(0, 0) == np.float64(-0.6666666666666666)

def test_photoreceptor_conversion_rate():
    """Test photoreceptor conversion rate calculation function."""
    # Test normal case
    assert photoreceptor_conversion_rate(1000, 1, '2018') == np.float64(0.0005019488349473618)
    assert photoreceptor_conversion_rate(100, 1, '2018') == np.float64(1.7361111111111114e-05)
    assert photoreceptor_conversion_rate(1, 1, '2018') == np.float64(1.7542013121425816e-08)
    assert photoreceptor_conversion_rate(1000, 0, '2018') == np.float64(0.0)
    assert photoreceptor_conversion_rate(100, 0, '2018') == np.float64(0.0)
    assert photoreceptor_conversion_rate(1, 0, '2018') == np.float64(0.0)
    
    # Test edge case with zero inputs
    assert photoreceptor_conversion_rate(0, 1, '2018') == np.float64(0.0)
    assert photoreceptor_conversion_rate(0, 0, '2018') == np.float64(0.0)

def test_photic_drive():
    """Test photic drive calculation function."""
    # Test normal case
    assert photic_drive(-0.14, -1.07, 0.10, 0.00016) == 0.000217147392
    assert photic_drive(0.25, -1.07, 0.10, 0.00016) == 0.0001850688
    assert photic_drive(-0.14, 1.2, 0.10, 0.00016) == 7.907328000000001e-05

def test_circadian_phase():
    """Test circadian phase calculation function."""
    # Test normal case
    assert circadian_phase(-0.14, -1.07) == np.float64(1.4406942690849316)


# def test_mean_population_firing_rate():
#     """Test mean population firing rate calculation function."""
#     # Test normal case
#     assert mean_population_firing_rate(0.1, 0.2, 0.3) == np.float64(0.1)
#     assert mean_population_firing_rate(0.5, 0.5, 0.5) == np.float64(0.5)
    
#     # Test edge case with zero inputs
#     assert mean_population_firing_rate(0, 0, 0) == np.float64(0.0)