from src.simulation import generate_spatial_temperature_data
from src.calibration import calibrate

def test_calibration():
    raw = generate_spatial_temperature_data()
    calibrated = calibrate(raw)

    assert calibrated["temperature"].std() > 0