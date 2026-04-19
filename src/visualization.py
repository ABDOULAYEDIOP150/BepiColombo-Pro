import matplotlib.pyplot as plt

def compare_signals(raw, filtered, calibrated, validated):

    # 📊 FIGURE 1 : Raw vs Filtered
    plt.figure(figsize=(10, 4))
    plt.plot(raw, label="Raw (brut)", alpha=0.7)
    plt.plot(filtered, label="Filtered (Butterworth)", alpha=0.9)
    plt.title("Raw vs Filtered")
    plt.xlabel("Index temporel (flatten)")
    plt.ylabel("Température")
    plt.legend()
    plt.grid()
    plt.show()

    # ⚙️ FIGURE 2 : Filtered vs Calibrated
    plt.figure(figsize=(10, 4))
    plt.plot(filtered, label="Filtered", alpha=0.7)
    plt.plot(calibrated, label="Calibrated", alpha=0.9)
    plt.title("Filtered vs Calibrated")
    plt.xlabel("Index temporel (flatten)")
    plt.ylabel("Température")
    plt.legend()
    plt.grid()
    plt.show()

    # 🧪 FIGURE 3 : Calibrated vs Validated
    plt.figure(figsize=(10, 4))
    plt.plot(calibrated, label="Calibrated", alpha=0.7)
    plt.plot(validated, label="Validated", linewidth=2)
    plt.title("Calibrated vs Validated (final dataset)")
    plt.xlabel("Index temporel (flatten)")
    plt.ylabel("Température")
    plt.legend()
    plt.grid()
    plt.show()

    # 📊 FIGURE 4 : Vue globale propre (option contrôle qualité)
    plt.figure(figsize=(10, 4))
    plt.plot(raw, label="Raw", alpha=0.3)
    plt.plot(filtered, label="Filtered", alpha=0.6)
    plt.plot(calibrated, label="Calibrated", alpha=0.8)
    plt.plot(validated, label="Validated", linewidth=2)
    plt.title("Overview (all stages)")
    plt.xlabel("Index temporel")
    plt.ylabel("Température")
    plt.legend()
    plt.grid()
    plt.show()
    