from src.simulation import generate_spatial_temperature_data
from src.calibration import calibrate
from src.validation import validate
from src.comparison import compare
from src.filtering import butterworth_filter
from src.visualization import compare_signals

def run_pipeline():

    print("🚀 Simulation données spatiales...")
    raw_ds = generate_spatial_temperature_data()

    raw = raw_ds["temperature"].values.reshape(-1)

    print("📡 Filtrage Butterworth...")
    filtered = butterworth_filter(raw)

    print("⚙️ Calibration instrumentale...")
    calibrated_ds = calibrate(raw_ds)
    calibrated = calibrated_ds["temperature"].values.reshape(-1)

    print("🧪 Validation scientifique...")
    validation = validate(calibrated_ds)

    print("📊 Comparaison métriques...")
    metrics = compare(raw_ds, calibrated_ds)

    # 🧪 signal "validé" = ici on prend calibré (version scientifique corrigée)
    validated = calibrated

    print("📊 Visualisation ingénieur complète...")
    compare_signals(raw, filtered, calibrated, validated)

    return {
        "validation": validation,
        "metrics": metrics
    }