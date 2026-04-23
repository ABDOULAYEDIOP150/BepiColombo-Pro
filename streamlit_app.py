"""
Streamlit dashboard for BepiColombo DBSC mission
1) Données brutes
2) FFT
3) Filtrage : Butterworth / Savitzky-Golay / Combo
4) Calibration et choix du meilleur modèle
5) Validation
6) Comparaison DBSC vs MAG
"""

import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import detrend, find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error

from simulation import generate_realistic_instrument_data
from filtering import (
    butterworth_filter,
    compare_to_true_signal,
    analyze_filter_effectiveness,
    grid_search_butterworth_parameters,
    grid_search_savgol_parameters,
    grid_search_combo_parameters,
    summarize_grid_search_results,
    plot_filter_and_fft,
    plot_filter_vs_true,
    plot_filter_benchmark,
)
from calibration import (
    calibrate_filtered_signal,
    inject_calibrated_signal_in_dataset,
    benchmark_calibration_models,
    summarize_calibration_benchmark,
    prove_calibration_quality,
)

# =============================================================================
# CONFIGURATION PAGE
# =============================================================================
st.set_page_config(
    page_title="BepiColombo DBSC - Réaliste",
    page_icon="🛰️",
    layout="wide"
)
# Chargement du thème CSS personnalisé
def load_css(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Appeler la fonction avec le chemin vers votre fichier CSS
load_css("style/theme.css")  # ou "assets/style.css" selon l'emplacement
# IMPORTANT :
# Le chargement du fichier style/theme.css a été désactivé ici car il provoque
# très probablement les chevauchements visuels sur les expanders
# (affichage de "arrow_right", "arrow_down", etc.).
# Si tu veux le réutiliser plus tard, il faudra corriger ce CSS.
#
# try:
#     with open("style/theme.css", "r", encoding="utf-8") as f:
#         css = f.read()
#         st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# except FileNotFoundError:
#     st.warning("Fichier style/theme.css non trouvé. Utilisation du thème par défaut.")
# except Exception as e:
#     st.warning(f"Erreur lors du chargement du CSS : {e}")

st.markdown("""
<style>
    .main {
        max-width: 1450px;
        margin: auto;
    }

    .stMarkdown, .stText, .stDataFrame, .stMetric {
        font-size: 18px !important;
    }

    h1 { font-size: 2.8rem !important; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }

    /* Sécurisation légère de l'affichage des expanders */
    details summary {
        white-space: normal !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ BepiColombo DBSC - Pipeline de validation réaliste")
st.markdown("### *De la donnée brute à la validation scientifique*")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================
def get_spatial_mean(ds):
    return ds["temperature"].mean(dim=["latitude", "longitude"]).values


def interpolate_nans(signal):
    signal = np.asarray(signal, dtype=float)
    mask_nan = np.isnan(signal)

    if np.all(mask_nan):
        return signal.copy(), mask_nan

    if np.any(mask_nan):
        x = np.arange(len(signal))
        signal_interp = np.interp(x, x[~mask_nan], signal[~mask_nan])
    else:
        signal_interp = signal.copy()

    return signal_interp, mask_nan


def compute_fft(signal, fs=1.0, detrend_signal=True):
    signal = np.asarray(signal, dtype=float)
    signal_interp, _ = interpolate_nans(signal)

    if np.all(np.isnan(signal_interp)):
        return np.array([]), np.array([])

    if detrend_signal:
        signal_interp = detrend(signal_interp)

    n = len(signal_interp)
    fft_vals = np.fft.rfft(signal_interp)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


def get_dominant_frequencies(freqs, magnitude, n_peaks=10, min_relative_height=0.1):
    if len(freqs) == 0 or len(magnitude) == 0 or len(magnitude) < 3:
        return []

    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * min_relative_height)

    if len(peaks) == 0:
        if len(magnitude) > 1:
            idx_sorted = np.argsort(magnitude[1:])[::-1] + 1
        else:
            idx_sorted = np.array([0])
    else:
        idx_sorted = peaks[np.argsort(magnitude[peaks])[::-1]]

    idx_sorted = idx_sorted[:n_peaks]

    results = []
    for idx in idx_sorted:
        f = freqs[idx]
        mag = magnitude[idx]
        period_s = np.inf if f == 0 else 1.0 / f
        period_h = np.inf if f == 0 else period_s / 3600.0
        results.append({
            "frequency_hz": float(f),
            "magnitude": float(mag),
            "period_s": float(period_s),
            "period_h": float(period_h)
        })
    return results


def count_frequencies_below_cutoff(freqs, cutoff):
    return int(np.sum(freqs <= cutoff))


def validate_ds(ds):
    data = ds["temperature"].values
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)
    valid_range = (min_val >= -50) and (max_val <= 100)

    return {
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean_val),
        "std": float(std_val),
        "valid_range": bool(valid_range)
    }


def compare_instruments(ds1, ds2):
    sig1 = ds1["temperature"].mean(dim=["latitude", "longitude"]).values
    sig2 = ds2["temperature"].mean(dim=["latitude", "longitude"]).values

    mask = ~(np.isnan(sig1) | np.isnan(sig2))
    if np.sum(mask) < 3:
        return {"MAE": np.nan, "RMSE": np.nan, "correlation": np.nan}

    mae = mean_absolute_error(sig1[mask], sig2[mask])
    rmse = np.sqrt(mean_squared_error(sig1[mask], sig2[mask]))
    corr = np.corrcoef(sig1[mask], sig2[mask])[0, 1]

    return {"MAE": float(mae), "RMSE": float(rmse), "correlation": float(corr)}


# =============================================================================
# FIGURES
# =============================================================================
def plot_raw_components(ds):
    time = ds["time_hours"].values
    raw = ds["temperature"].mean(dim=["latitude", "longitude"]).values
    phys = ds["physical_signal"].values
    inst_temp = ds["instrument_temperature"].values

    raw_filled = np.nan_to_num(raw, nan=np.nanmean(raw))
    smooth = uniform_filter1d(raw_filled, size=20, mode="nearest")
    noise_est = raw_filled - smooth
    drift_est = raw_filled - phys - np.nanmedian(raw_filled - phys)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    axes[0, 0].plot(time, raw, linewidth=1.0)
    axes[0, 0].set_title("Signal brut mesuré")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].plot(time, phys, color="green", linewidth=1.2)
    axes[0, 1].set_title("Signal physique")
    axes[0, 1].grid(True)

    axes[1, 0].plot(time, drift_est, color="red", linewidth=1.0, label="Dérive estimée")
    axes[1, 0].plot(time, inst_temp, color="orange", linewidth=1.0, alpha=0.8, label="Température instrument")
    axes[1, 0].set_title("Dérive estimée / température instrumentale")
    axes[1, 0].set_xlabel("Temps [heures]")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(time, noise_est, color="purple", linewidth=1.0, label="Bruit HF estimé")
    axes[1, 1].set_title("Bruit haute fréquence estimé")
    axes[1, 1].set_xlabel("Temps [heures]")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig


def plot_fft_before(raw_signal, fs, x_max):
    freqs, mag = compute_fft(raw_signal, fs=fs, detrend_signal=True)
    dominant = get_dominant_frequencies(freqs, mag, n_peaks=10, min_relative_height=0.1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(freqs, mag, linewidth=1.5, label="FFT signal brut")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("Magnitude (log)")
    ax.set_title("FFT du signal brut")
    ax.grid(True, alpha=0.3)

    for item in dominant[:5]:
        ax.axvline(item["frequency_hz"], linestyle="--", alpha=0.4)

    ax.legend()
    plt.tight_layout()
    return fig, freqs, mag, dominant


# =============================================================================
# SESSION STATE
# =============================================================================
defaults = {
    "data_loaded": False,
    "dbsc_raw": None,
    "mag_raw": None,
    "dbsc_clean_reference": None,
    "dbsc_filter_models": None,
    "dbsc_best_filter_model_name": None,
    "dbsc_best_filter_result": None,
    "dbsc_calibration_results": None,
    "dbsc_best_calibrated_model_name": None,
    "dbsc_best_calibrated_signal": None,
    "dbsc_calibrated": None,
    "mag_calibrated": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("🛰️ 1. Génération des données")

duration_hours = st.sidebar.slider(
    "Durée (heures)",
    min_value=1,
    max_value=120,
    value=48,
    step=1
)

sampling_rate = st.sidebar.selectbox(
    "Fréquence d'échantillonnage (Hz)",
    [0.5, 1.0, 2.0],
    index=1
)

include_gaps = st.sidebar.checkbox("Inclure gaps (NaN)", value=True)
include_glitches = st.sidebar.checkbox("Inclure glitches", value=True)

with st.sidebar.expander("Aide - fréquence d'échantillonnage"):
    st.write("0.5 Hz = 1 mesure toutes les 2 secondes")
    st.write("1.0 Hz = 1 mesure par seconde")
    st.write("2.0 Hz = 2 mesures par seconde")

if st.sidebar.button("🚀 Générer DBSC & MAG", type="primary"):
    with st.spinner("Génération des données réalistes..."):
        st.session_state.dbsc_raw = generate_realistic_instrument_data(
            duration_hours=duration_hours,
            sampling_rate_hz=sampling_rate,
            instrument_name="DBSC",
            seed=42,
            include_gaps=include_gaps,
            include_glitches=include_glitches,
            include_saturation=False
        )

        st.session_state.mag_raw = generate_realistic_instrument_data(
            duration_hours=duration_hours,
            sampling_rate_hz=sampling_rate,
            instrument_name="MAG",
            seed=123,
            include_gaps=include_gaps,
            include_glitches=include_glitches,
            include_saturation=False
        )

        st.session_state.dbsc_clean_reference = generate_realistic_instrument_data(
            duration_hours=duration_hours,
            sampling_rate_hz=sampling_rate,
            instrument_name="DBSC",
            seed=42,
            include_gaps=False,
            include_glitches=False,
            include_saturation=False
        )

        st.session_state.dbsc_filter_models = None
        st.session_state.dbsc_best_filter_model_name = None
        st.session_state.dbsc_best_filter_result = None
        st.session_state.dbsc_calibration_results = None
        st.session_state.dbsc_best_calibrated_model_name = None
        st.session_state.dbsc_best_calibrated_signal = None
        st.session_state.dbsc_calibrated = None
        st.session_state.mag_calibrated = None
        st.session_state.data_loaded = True

    st.sidebar.success("✅ Données générées")

st.sidebar.markdown("---")
menu_step = st.sidebar.radio(
    "Choisir l'étape :",
    [
        "1️⃣ DONNÉES BRUTES - Composantes",
        "2️⃣ FFT - Analyse fréquentielle",
        "3️⃣ FILTRAGE - Benchmark des modèles",
        "4️⃣ CALIBRATION - Benchmark final",
        "5️⃣ VALIDATION - Normes ESA/JAXA",
        "6️⃣ COMPARAISON DBSC vs MAG"
    ]
)

if not st.session_state.data_loaded:
    st.info("👈 Configure puis génère les données dans la barre latérale.")
    st.stop()


# =============================================================================
# PARTIE 1
# =============================================================================
if menu_step == "1️⃣ DONNÉES BRUTES - Composantes":
    st.header("1️⃣ Données brutes - composantes du signal")

    st.markdown(
        """
Cette première étape présente la structure des données brutes utilisées dans le projet.

L'objectif est de montrer qu'un signal capteur industriel n'est pas un signal pur : il contient à la fois
une composante physique utile, du bruit, des artefacts ponctuels, des données manquantes et une dérive
instrumentale lente. Avant toute correction, il est donc nécessaire de comprendre la nature du signal observé.
"""
    )

    st.subheader("Structure du dataset brut")

    st.write(
        "Le dataset généré est un dataset multidimensionnel de type xarray. "
        "Il contient le signal mesuré, le signal physique de référence simulé, "
        "la température instrumentale ainsi que des indicateurs de qualité."
    )

    with st.expander("Afficher la structure xarray du dataset DBSC"):
        st.code(repr(st.session_state.dbsc_raw), language="python")

    st.subheader("Comparaison avec le dataset de référence")

    st.write(
        "Le dataset de référence correspond à une version sans glitches et sans données manquantes. "
        "Le tableau ci-dessous montre quelques instants où le signal généré diffère du signal de référence."
    )

    clean = st.session_state.dbsc_clean_reference["temperature"][:, 0, 0].values
    raw = st.session_state.dbsc_raw["temperature"][:, 0, 0].values

    time_vals = st.session_state.dbsc_raw["time"].values
    time_hours = st.session_state.dbsc_raw["time_hours"].values
    phys_vals = st.session_state.dbsc_raw["physical_signal"].values
    inst_temp_vals = st.session_state.dbsc_raw["instrument_temperature"].values
    cal_flag_vals = st.session_state.dbsc_raw["calibration_flag"].values

    diff_mask = (
        (np.abs(clean - raw) > 1e-6) |
        np.isnan(raw) |
        np.isnan(clean)
    )
    diff_indices = np.where(diff_mask)[0]

    if len(diff_indices) == 0:
        st.success("Aucune différence détectée entre le dataset de référence et le dataset généré.")
    else:
        diff_indices = diff_indices[:5]

        comparison_df = pd.DataFrame({
            "time": time_vals[diff_indices],
            "time_hours": time_hours[diff_indices],
            "temperature_reference": clean[diff_indices],
            "temperature_generee": raw[diff_indices],
            "difference": raw[diff_indices] - clean[diff_indices],
            "physical_signal": phys_vals[diff_indices],
            "instrument_temperature": inst_temp_vals[diff_indices],
            "calibration_flag": cal_flag_vals[diff_indices]
        })

        st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Indicateurs de qualité des données")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "DBSC - Données manquantes",
            f"{st.session_state.dbsc_raw.attrs['quality_flags']['missing_data_percent']:.2f}%"
        )
        st.metric(
            "DBSC - Glitches",
            st.session_state.dbsc_raw.attrs["quality_flags"]["glitches_count"]
        )

    with col2:
        st.metric(
            "MAG - Données manquantes",
            f"{st.session_state.mag_raw.attrs['quality_flags']['missing_data_percent']:.2f}%"
        )
        st.metric(
            "MAG - Glitches",
            st.session_state.mag_raw.attrs["quality_flags"]["glitches_count"]
        )

    st.subheader("Décomposition visuelle du signal brut")
    st.pyplot(plot_raw_components(st.session_state.dbsc_raw))

    st.subheader("Interprétation")

    st.markdown(
        """
Les figures précédentes montrent que la donnée brute n'est pas directement exploitable telle quelle.

- **Signal brut mesuré** : il représente la mesure réellement fournie par l'instrument.  
  Ce signal contient la dynamique utile, mais aussi les perturbations dues au système de mesure.

- **Signal physique** : il correspond à la composante utile du phénomène simulé.  
  C'est cette information que l'on cherche à retrouver au mieux après traitement.

- **Dérive estimée / température instrumentale** : on observe une composante lente qui ne relève pas du phénomène physique lui-même, mais du comportement de l'instrument au cours du temps.

- **Bruit haute fréquence estimé** : cette composante traduit la présence de fluctuations rapides qui perturbent la lecture du signal utile.  
  À cela peuvent s'ajouter des points aberrants ponctuels, appelés glitches, ainsi que des valeurs manquantes.
"""
    )

    st.subheader("Conclusion de l'étape")

    st.markdown(
        """
Le but de ce projet est de simuler un cas réaliste de traitement de données industrielles issues de capteurs.

Dans ce type de contexte, le signal observé est généralement un **signal mélangé** :
- une composante physique utile,
- du bruit,
- des artefacts ponctuels,
- des données manquantes,
- et une dérive instrumentale.

L'enjeu du pipeline est donc d'identifier ces composantes, de comprendre leurs effets sur la mesure, puis de proposer un traitement adapté pour rapprocher le signal mesuré du signal physique réel.

Pour analyser plus finement la structure fréquentielle de cette série temporelle, l'étape suivante consiste à utiliser la **transformée de Fourier**, afin d'identifier les fréquences dominantes présentes dans les données avant filtrage.
"""
    )

    with st.expander("Résumé technique de cette étape"):
        st.markdown(
            """
- **Variable mesurée principale** : `temperature`
- **Signal physique simulé** : `physical_signal`
- **Température instrumentale** : `instrument_temperature`
- **Indicateur qualité** : `calibration_flag`
- **Référence propre** : dataset sans glitches et sans NaN
- **Objectif de l'étape** : caractériser la donnée brute avant toute opération de filtrage et de calibration
"""
        )

# =============================================================================
# PARTIE 2
# =============================================================================
elif menu_step == "2️⃣ FFT - Analyse fréquentielle":
    st.header("2️⃣ FFT - analyse fréquentielle du signal brut")

    raw_signal = get_spatial_mean(st.session_state.dbsc_raw)
    fs = st.session_state.dbsc_raw.attrs["sampling_rate_hz"]

    st.markdown(
        """
Cette étape consiste à analyser le contenu fréquentiel du signal brut à l'aide de la transformée de Fourier.

L'objectif n'est pas encore de filtrer, mais de comprendre la structure du signal :
- identifier les fréquences dominantes présentes dans la mesure,
- vérifier si le signal contient plusieurs composantes fréquentielles,
- repérer la présence d'un fond de bruit,
- et proposer une fréquence de coupure cohérente pour l'étape de filtrage.

La FFT permet donc de déterminer si le signal observé est un signal simple ou un signal mélangé.
Dans notre cas, cette analyse est essentielle pour préparer le choix des paramètres du filtrage.
"""
    )

    x_max_default = min(0.002, fs / 2.0)
    x_max = st.slider(
        "Zoom fréquence max (Hz)",
        min_value=0.0005,
        max_value=float(min(0.01, fs / 2.0)),
        value=float(x_max_default),
        step=0.0005,
        format="%.4f"
    )

    min_relative_height = st.slider(
        "Seuil relatif de détection des pics FFT",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f"
    )

    max_reported_peaks = st.slider(
        "Nombre maximal de pics à afficher",
        min_value=3,
        max_value=10,
        value=6,
        step=1
    )

    fig, freqs, mag, _ = plot_fft_before(raw_signal, fs=fs, x_max=x_max)
    st.pyplot(fig)

    dominant = get_dominant_frequencies(
        freqs,
        mag,
        n_peaks=max_reported_peaks,
        min_relative_height=min_relative_height
    )

    cutoff_preview = st.number_input(
        "Fréquence de coupure envisagée pour le filtrage (Hz)",
        min_value=0.0001,
        max_value=float(fs / 2.0 - 1e-6),
        value=min(0.0015, float(fs / 2.0 - 1e-6)),
        step=0.0001,
        format="%.4f"
    )

    n_below = count_frequencies_below_cutoff(freqs, cutoff_preview)

    st.subheader("Résultats FFT")

    st.write(f"**Nombre total de fréquences dans le spectre** : {len(freqs)}")
    st.write(f"**Nombre de fréquences sous {cutoff_preview:.4f} Hz** : {n_below}")
    st.write(f"**Nombre de pics dominants détectés avec le seuil choisi** : {len(dominant)}")

    if len(dominant) == 0:
        st.warning(
            "Aucun pic dominant n'a été détecté avec ce seuil. "
            "Tu peux diminuer le seuil relatif pour rendre la détection plus sensible."
        )
    else:
        df = pd.DataFrame(dominant)
        st.subheader("Fréquences dominantes détectées")
        st.dataframe(df, use_container_width=True)

        max_dominant_freq = max(item["frequency_hz"] for item in dominant)
        suggested_cutoff = 1.2 * max_dominant_freq

        st.subheader("Interprétation")

        st.markdown(
            f"""
L'analyse fréquentielle montre que le signal n'est pas constitué d'une seule fréquence, mais de **plusieurs composantes dominantes**.

Le nombre de pics détectés ici est de **{len(dominant)}**, mais cette valeur n'est pas fixe :
elle dépend :
- du signal observé,
- de la durée de la simulation,
- de la fréquence d'échantillonnage,
- et du seuil choisi pour détecter les pics.

Cette étape met donc en évidence que le signal mesuré est un **signal mélangé** :
- il contient plusieurs composantes fréquentielles utiles,
- mais aussi un fond spectral plus diffus, qui traduit la présence de bruit.

La fréquence dominante utile la plus élevée détectée dans cette configuration est :

- **f_max dominante = {max_dominant_freq:.6f} Hz**

À partir de cette valeur, une fréquence de coupure initiale peut être proposée pour un filtre passe-bas :

- **f_c proposée ≈ {suggested_cutoff:.6f} Hz**

L'idée est de choisir une coupure légèrement supérieure à la fréquence utile la plus élevée,
afin de conserver les composantes principales du signal tout en atténuant les composantes plus rapides,
généralement associées au bruit.
"""
        )

    st.info(
        "Conclusion : la FFT permet de caractériser le signal brut, de montrer qu'il s'agit d'un signal composé de plusieurs fréquences, et de guider le choix initial de la fréquence de coupure avant filtrage."
    )

    with st.expander("Résumé technique de cette étape"):
        st.markdown(
            """
- **Outil utilisé** : transformée de Fourier (FFT)
- **Objectif** : analyser le contenu fréquentiel du signal brut
- **Ce qu'on cherche** :
  - les fréquences dominantes,
  - la présence éventuelle de bruit,
  - une fréquence de coupure cohérente pour le filtrage
- **Point important** : le nombre de fréquences dominantes détectées n'est pas constant ; il dépend du signal et du seuil de détection utilisé
"""
        )

# =============================================================================
# PARTIE 3
# =============================================================================
elif menu_step == "3️⃣ FILTRAGE - Benchmark des modèles":
    st.header("3️⃣ Benchmark des modèles de filtrage")

    raw_signal = get_spatial_mean(st.session_state.dbsc_raw)
    true_signal = st.session_state.dbsc_raw["physical_signal"].values
    fs = st.session_state.dbsc_raw.attrs["sampling_rate_hz"]

    st.markdown(
        """
Cette étape a pour objectif de comparer plusieurs stratégies de filtrage afin d'identifier celle
qui reconstruit le mieux le signal physique à partir du signal brut.

Trois approches sont testées :

- **Butterworth** : filtre passe-bas classique, adapté lorsque l'analyse FFT suggère qu'il faut conserver
  les basses fréquences utiles et atténuer les composantes rapides.
- **Savitzky-Golay** : lissage polynomial local, utile pour lisser le signal tout en conservant la forme
  générale des variations.
- **Combo** : application successive de Butterworth puis de Savitzky-Golay, afin de combiner réduction
  fréquentielle du bruit et lissage local.

Chaque modèle est évalué par validation croisée sur plusieurs jeux de paramètres.  
Le meilleur modèle est ensuite retenu à partir des métriques de comparaison au signal réel.
"""
    )

    st.subheader("Principe de choix des paramètres")

    st.markdown(
        """
Le choix des paramètres n'est pas arbitraire.

Pour **Butterworth**, on s'appuie sur l'analyse FFT de l'étape précédente :
- la fréquence dominante utile la plus élevée détectée dans le signal donne une référence,
- la fréquence de coupure testée doit être légèrement supérieure à cette valeur,
- plusieurs couples *(ordre, coupure)* sont ensuite évalués par validation croisée.

Pour **Savitzky-Golay**, les paramètres testés sont :
- la **taille de fenêtre**,
- l'**ordre polynomial**.

Pour le **Combo**, on applique d'abord Butterworth puis Savitzky-Golay, et on teste plusieurs combinaisons
des paramètres des deux méthodes.
"""
    )

    max_dominant_freq_input = st.number_input(
        "Fréquence dominante maximale retenue depuis la FFT (Hz)",
        min_value=0.00001,
        max_value=float(fs / 2.0 - 1e-6),
        value=0.0010,
        step=0.0001,
        format="%.5f"
    )

    st.caption(
        "Cette valeur provient de l'étape FFT. Elle sert de référence pour définir l'intervalle des "
        "fréquences de coupure testées pour Butterworth et pour le modèle Combo."
    )

    st.subheader("A. Validation croisée Butterworth")

    st.markdown(
        """
Le filtre Butterworth est un filtre passe-bas.  
Son rôle ici est de supprimer les composantes haute fréquence associées au bruit, tout en conservant
les fréquences dominantes du signal utile.

Les paramètres évalués sont :
- **l'ordre du filtre** : il contrôle la pente de coupure,
- **la fréquence de coupure** : elle doit rester légèrement supérieure à la fréquence utile maximale
  détectée dans la FFT.
"""
    )

    n_cutoffs = st.slider(
        "Nombre de coupures Butterworth testées",
        min_value=4,
        max_value=20,
        value=8,
        step=1
    )

    st.subheader("B. Validation croisée Savitzky-Golay")

    st.markdown(
        """
Le filtre Savitzky-Golay réalise un lissage local par ajustement polynomial.
Il est utile pour atténuer le bruit tout en conservant la forme des oscillations du signal.

Les paramètres évalués sont :
- **la taille de fenêtre**,
- **l'ordre polynomial**.
"""
    )

    sg_windows = st.multiselect(
        "Fenêtres Savitzky-Golay",
        [11, 21, 31, 41, 51],
        default=[11, 21, 31]
    )

    sg_polyorders = st.multiselect(
        "Ordres polynomiaux Savitzky-Golay",
        [2, 3, 4],
        default=[2, 3]
    )

    st.subheader("C. Validation croisée Combo")

    st.markdown(
        """
Le modèle Combo applique successivement :
1. un filtrage Butterworth,
2. puis un lissage Savitzky-Golay.

Cette stratégie cherche à profiter des deux approches :
- Butterworth pour contrôler la coupure fréquentielle,
- Savitzky-Golay pour améliorer le lissage local.
"""
    )

    combo_windows = st.multiselect(
        "Fenêtres Combo (Savitzky-Golay après Butterworth)",
        [11, 21, 31, 41],
        default=[11, 21]
    )

    combo_polyorders = st.multiselect(
        "Ordres polynomiaux Combo",
        [2, 3, 4],
        default=[2, 3]
    )

    if len(sg_windows) == 0 or len(sg_polyorders) == 0:
        st.warning("Sélectionne au moins une fenêtre et un ordre polynomial pour Savitzky-Golay.")
        st.stop()

    if len(combo_windows) == 0 or len(combo_polyorders) == 0:
        st.warning("Sélectionne au moins une fenêtre et un ordre polynomial pour le modèle Combo.")
        st.stop()

    best_butter, all_butter = grid_search_butterworth_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        fs=fs,
        max_dominant_freq=max_dominant_freq_input,
        orders=(3, 4, 5, 6),
        n_cutoffs=n_cutoffs,
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.5,
    )

    best_sg, all_sg = grid_search_savgol_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        window_lengths=tuple(sorted(sg_windows)),
        polyorders=tuple(sorted(sg_polyorders)),
    )

    best_combo, all_combo = grid_search_combo_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        fs=fs,
        max_dominant_freq=max_dominant_freq_input,
        butter_orders=(3, 4, 5, 6),
        n_cutoffs=max(4, min(8, n_cutoffs)),
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.5,
        window_lengths=tuple(sorted(combo_windows)),
        polyorders=tuple(sorted(combo_polyorders)),
    )

    filter_models = {
        "Butterworth": best_butter,
        "Savitzky-Golay": best_sg,
        "Combo": best_combo,
    }

    best_filter_model_name = max(filter_models.keys(), key=lambda k: filter_models[k]["score"])
    best_filter_result = filter_models[best_filter_model_name]

    st.session_state.dbsc_filter_models = filter_models
    st.session_state.dbsc_best_filter_model_name = best_filter_model_name
    st.session_state.dbsc_best_filter_result = best_filter_result

    filtered_signals = {
        name: result["filtered_signal"]
        for name, result in filter_models.items()
    }

    st.subheader("D. Résumé des meilleurs modèles")

    benchmark_df = pd.DataFrame([
        {
            "model": name,
            "score": result["score"],
            "mae": result["filtered_vs_true"]["MAE"],
            "rmse": result["filtered_vs_true"]["RMSE"],
            "corr": result["filtered_vs_true"]["correlation"],
            "gain_mae": result["gain_mae"],
            "gain_rmse": result["gain_rmse"],
            "gain_corr": result["gain_corr"],
        }
        for name, result in filter_models.items()
    ]).sort_values("score", ascending=False)

    st.dataframe(benchmark_df, use_container_width=True)

    st.subheader("E. Meilleur modèle retenu")

    m1, m2 = st.columns(2)
    m1.metric("Modèle optimal", best_filter_model_name)
    m2.metric("Score optimal", f"{best_filter_result['score']:.4f}")

    st.markdown(
        f"""
Le meilleur modèle retenu à cette étape est **{best_filter_model_name}**.

Le score utilisé prend en compte :
- l'amélioration de la **MAE**,
- l'amélioration de la **RMSE**,
- l'amélioration de la **corrélation** avec le signal réel,
- ainsi que la réduction du bruit lorsque cette information est disponible.

Ce choix permet de retenir le filtrage qui rapproche le plus le signal brut du signal physique simulé.
"""
    )

    st.subheader("F. Comparaison visuelle des modèles")
    st.pyplot(plot_filter_benchmark(raw_signal, true_signal, filtered_signals, fs))

    st.markdown(
        """
Cette figure compare les trois meilleurs signaux filtrés au signal brut et au signal réel.

Elle permet de vérifier visuellement :
- si le bruit a été réduit,
- si la forme générale du signal a été conservée,
- et si le signal filtré reste cohérent avec la dynamique du signal physique.

L'objectif n'est pas seulement de lisser le signal, mais de **réduire le bruit sans déformer
les composantes utiles**.
"""
    )

    st.subheader("G. Détail des validations croisées")

    tab1, tab2, tab3 = st.tabs(["Butterworth", "Savitzky-Golay", "Combo"])

    with tab1:
        st.markdown(
            """
Ici, chaque ligne correspond à un couple de paramètres Butterworth testé.
On compare les performances obtenues pour différentes fréquences de coupure et différents ordres.
"""
        )
        st.dataframe(
            pd.DataFrame(summarize_grid_search_results(all_butter)).sort_values("score", ascending=False),
            use_container_width=True
        )

    with tab2:
        st.markdown(
            """
Ici, chaque ligne correspond à une combinaison *(fenêtre, ordre polynomial)* testée
pour Savitzky-Golay.
"""
        )
        st.dataframe(
            pd.DataFrame(summarize_grid_search_results(all_sg)).sort_values("score", ascending=False),
            use_container_width=True
        )

    with tab3:
        st.markdown(
            """
Ici, chaque ligne correspond à une combinaison de paramètres du modèle hybride :
Butterworth suivi de Savitzky-Golay.
"""
        )
        st.dataframe(
            pd.DataFrame(summarize_grid_search_results(all_combo)).sort_values("score", ascending=False),
            use_container_width=True
        )

    st.subheader("H. Analyse détaillée du meilleur filtrage")

    freqs_raw, mag_raw = compute_fft(raw_signal, fs=fs, detrend_signal=True)
    freqs_best, mag_best = compute_fft(best_filter_result["filtered_signal"], fs=fs, detrend_signal=True)
    x_max = min(0.002, fs / 2.0)

    cutoff_to_show = best_filter_result.get("cutoff", None)
    title_prefix = f"{best_filter_model_name} - "

    st.pyplot(
        plot_filter_and_fft(
            raw_signal,
            best_filter_result["filtered_signal"],
            freqs_raw,
            mag_raw,
            freqs_best,
            mag_best,
            fs=fs,
            cutoff=cutoff_to_show,
            x_max=x_max,
            title_prefix=title_prefix
        )
    )

    st.markdown(
        """
Cette figure détaille le comportement du meilleur modèle retenu.

Elle contient quatre lectures complémentaires :

1. **Signal temporel brut vs filtré**  
   Cette vue permet de voir si le bruit a été atténué sans dégrader la structure globale du signal.

2. **Zoom temporel**  
   Ce zoom permet de vérifier localement si le filtrage reste cohérent et s'il ne supprime pas
   les variations utiles du signal.

3. **FFT avant filtrage**  
   On observe la structure fréquentielle initiale du signal brut, avec les composantes utiles
   et le fond de bruit.

4. **FFT après filtrage**  
   On vérifie ici que les composantes de bruit situées au-delà de la coupure ont été atténuées,
   tout en conservant les fréquences dominantes utiles du signal.

Dans le cas de Butterworth, la ligne verticale de coupure permet d'interpréter directement
la séparation entre les fréquences conservées et celles qui sont atténuées.
"""
    )

    st.subheader("I. Comparaison du meilleur filtrage au signal réel")

    st.pyplot(
        plot_filter_vs_true(
            raw_signal=raw_signal,
            filtered_signal=best_filter_result["filtered_signal"],
            true_signal=true_signal,
            fs=fs,
            model_name=best_filter_model_name
        )
    )

    st.markdown(
        """
Cette figure permet de juger directement la qualité du filtrage par rapport au signal réel simulé.

- La première partie montre si le signal filtré se rapproche visuellement du signal utile.
- La seconde partie montre les **écarts au signal réel** avant et après filtrage.

Si l'écart filtré devient plus faible que l'écart brut, alors le filtrage a bien amélioré
la reconstruction du signal physique.
"""
    )

    st.subheader("Conclusion de l'étape")

    st.markdown(
        f"""
Cette étape montre que le filtrage ne doit pas être choisi arbitrairement.

- L'analyse FFT précédente sert à guider le choix initial des paramètres, en particulier pour Butterworth.
- La validation croisée permet ensuite de comparer objectivement plusieurs familles de filtres.
- Le meilleur modèle retenu ici est **{best_filter_model_name}**, car c'est celui qui offre le meilleur compromis
  entre réduction du bruit et conservation de l'information utile.

Le signal filtré retenu sera utilisé à l'étape suivante pour la calibration.
"""
    )

    with st.expander("Résumé technique de cette étape"):
        st.markdown(
            f"""
- **Méthodes comparées** : Butterworth, Savitzky-Golay, Combo
- **Signal de référence** : `physical_signal`
- **Critères de comparaison** : MAE, RMSE, corrélation, score global
- **Meilleur modèle retenu** : `{best_filter_model_name}`
- **Objectif atteint** : sélectionner le filtrage qui conserve le maximum d'information utile tout en réduisant le bruit
"""
        )

# =============================================================================
# PARTIE 4
# =============================================================================
elif menu_step == "4️⃣ CALIBRATION - Benchmark final":
    st.header("4️⃣ Calibration des modèles filtrés et choix final")

    if st.session_state.dbsc_filter_models is None:
        st.warning("Exécute d'abord l'étape 3 de filtrage.")
        st.stop()

    raw_signal = get_spatial_mean(st.session_state.dbsc_raw)
    true_signal = st.session_state.dbsc_raw["physical_signal"].values
    time_h = st.session_state.dbsc_raw["time_hours"].values

    filtered_models = {
        name: result["filtered_signal"]
        for name, result in st.session_state.dbsc_filter_models.items()
    }

    best_filter_model_name = st.session_state.dbsc_best_filter_model_name

    st.markdown(
        f"""
Cette étape applique la **calibration** aux signaux déjà filtrés à l'étape précédente.

Le principe est le suivant :

- on ne calibre pas directement le signal brut,
- on calibre les **meilleures versions filtrées** obtenues par chaque méthode,
- puis on compare les résultats calibrés afin de sélectionner le **meilleur modèle final**.

Autrement dit, la calibration s'appuie sur les estimateurs construits après filtrage.  
Le meilleur filtrage identifié à l'étape 3 est actuellement :

- **{best_filter_model_name}**

mais la calibration peut modifier le classement final.  
Il est donc nécessaire de réévaluer les modèles après retrait de dérive et correction instrumentale.
"""
    )

    st.subheader("Principe de la calibration")

    st.markdown(
        """
La calibration a pour objectif de corriger les effets instrumentaux lents qui restent présents
après filtrage.

Dans ce projet, la calibration repose sur :
- l'estimation d'une **dérive lente** au cours du temps,
- sa suppression par ajustement polynomial,
- puis l'application d'un **gain** et d'un **offset** instrumentaux.

Cette étape est importante car un signal bien filtré peut encore rester biaisé ou dériver
dans le temps, même si son bruit haute fréquence a été réduit.
"""
    )

    st.subheader("Paramètres de calibration")

    drift_degree = st.slider(
        "Degré du polynôme de dérive",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

    preserve_mean = st.checkbox(
        "Préserver la moyenne après retrait de dérive",
        value=True
    )

    st.markdown(
        """
- **Degré du polynôme de dérive** : il contrôle la complexité de la dérive modélisée.
  - degré 1 : dérive linéaire,
  - degré 2 ou plus : dérive plus souple.
- **Préserver la moyenne** : permet de corriger la dérive tout en gardant un niveau moyen cohérent.
"""
    )

    best_model_name, calibration_results = benchmark_calibration_models(
        filtered_models=filtered_models,
        raw_signal=raw_signal,
        true_signal=true_signal,
        time_1d=time_h,
        instrument=st.session_state.dbsc_raw.attrs.get("instrument", "DBSC"),
        drift_degree=drift_degree,
        preserve_mean=preserve_mean
    )

    best_calibrated_signal = calibration_results[best_model_name]["calibrated_signal"]

    st.session_state.dbsc_calibration_results = calibration_results
    st.session_state.dbsc_best_calibrated_model_name = best_model_name
    st.session_state.dbsc_best_calibrated_signal = best_calibrated_signal
    st.session_state.dbsc_calibrated = inject_calibrated_signal_in_dataset(
        st.session_state.dbsc_raw,
        best_calibrated_signal
    )

    st.subheader("Meilleur modèle final")

    c1, c2 = st.columns(2)
    c1.metric("Modèle final optimal", best_model_name)
    c2.metric("Score calibration", f"{calibration_results[best_model_name]['score']:.4f}")

    st.markdown(
        f"""
Le meilleur modèle après calibration est **{best_model_name}**.

Cela signifie que, parmi les modèles filtrés issus de l'étape précédente, c'est ce modèle calibré
qui présente le meilleur compromis entre :

- réduction de l'erreur au signal réel,
- amélioration de la corrélation,
- et suppression de la dérive résiduelle.

Le meilleur modèle après filtrage n'est donc pas nécessairement celui qui reste optimal après calibration.
Cette étape permet justement de vérifier cela de manière quantitative.
"""
    )

    st.subheader("Comparaison globale des modèles calibrés")

    calibrated_models = {
        model_name: result["calibrated_signal"]
        for model_name, result in calibration_results.items()
    }

    st.pyplot(
        prove_calibration_quality(
            raw_signal=raw_signal,
            true_signal=true_signal,
            calibrated_models=calibrated_models,
            time_1d=time_h
        )
    )

    st.markdown(
        """
Cette figure globale permet d'analyser la calibration sous plusieurs angles.

### 1. Comparaison globale des modèles calibrés
On compare ici :
- le signal brut,
- le signal réel,
- et les versions calibrées des différents modèles.

Cette vue permet de vérifier si les signaux calibrés se rapprochent du signal utile tout en conservant
une dynamique temporelle cohérente.

### 2. Écarts au signal réel
Cette figure est essentielle :
elle montre directement l'écart entre chaque signal calibré et le signal réel.

Une baisse de l'écart indique une amélioration effective.  
On cherche donc :
- un écart plus faible que celui du brut,
- une trajectoire plus stable,
- et une dispersion réduite.

### 3. Distribution des signaux calibrés
Cette représentation permet de comparer la répartition statistique des modèles calibrés.

Si les distributions sont proches, cela signifie que les modèles convergent vers une structure globale similaire.  
Si l'une des distributions s'écarte fortement, cela peut indiquer un biais résiduel.

### 4. Relation entre modèles calibrés
Le nuage de points compare directement les modèles calibrés entre eux.

S'ils sont fortement alignés, cela signifie qu'après calibration, les modèles deviennent très proches.  
Dans ce cas, la différence finale se joue surtout sur les métriques fines et la dérive résiduelle.
"""
    )

    calibration_df = pd.DataFrame(
        summarize_calibration_benchmark(calibration_results)
    ).sort_values("score", ascending=False)

    st.subheader("Résumé benchmark calibration")
    st.dataframe(calibration_df, use_container_width=True)

    st.markdown(
        """
Le tableau ci-dessus résume les performances finales des modèles calibrés.

On y lit notamment :
- le score global de calibration,
- la MAE après calibration,
- la RMSE après calibration,
- la corrélation après calibration,
- les gains obtenus par rapport au signal filtré,
- ainsi que la dérive supprimée.

Ce tableau permet de justifier objectivement le choix du modèle final.
"""
    )

    st.subheader("Détail des métriques par modèle")

    for model_name, result in calibration_results.items():
        with st.expander(f"Modèle {model_name}"):
            m = result["metrics"]

            st.markdown(
                f"""
Ce bloc compare le modèle **{model_name}** avant calibration et après calibration.

L'objectif est de vérifier si la calibration apporte :
- une réduction de l'erreur,
- une hausse de la corrélation,
- une réduction de la dérive.
"""
            )

            a1, a2, a3 = st.columns(3)
            a1.metric("MAE filtré → réel", f"{m['filtered_vs_true']['MAE']:.4f}")
            a2.metric("RMSE filtré → réel", f"{m['filtered_vs_true']['RMSE']:.4f}")
            a3.metric("Corr filtré → réel", f"{m['filtered_vs_true']['correlation']:.4f}")

            b1, b2, b3 = st.columns(3)
            b1.metric("MAE calibré → réel", f"{m['calibrated_vs_true']['MAE']:.4f}")
            b2.metric("RMSE calibré → réel", f"{m['calibrated_vs_true']['RMSE']:.4f}")
            b3.metric("Corr calibré → réel", f"{m['calibrated_vs_true']['correlation']:.4f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Gain MAE", f"{m['gain_mae_vs_filtered']:.4f}")
            d2.metric("Gain RMSE", f"{m['gain_rmse_vs_filtered']:.4f}")
            d3.metric("Gain corr", f"{m['gain_corr_vs_filtered']:.4f}")

            e1, e2, e3 = st.columns(3)
            e1.metric("Dérive filtré", f"{m['drift_filtered']:.4f}")
            e2.metric("Dérive calibré", f"{m['drift_calibrated']:.4f}")
            e3.metric("Dérive supprimée", f"{m['drift_removed']:.4f}")

            st.markdown(
                f"""
### Interprétation du modèle {model_name}

- **MAE filtré → réel** et **RMSE filtré → réel** : elles mesurent l'erreur avant calibration.
- **MAE calibré → réel** et **RMSE calibré → réel** : elles montrent si la calibration réduit cette erreur.
- **Corr calibré → réel** : elle indique si la structure temporelle du signal calibré reste cohérente
  avec le signal réel.
- **Gains** :
  - un gain MAE positif signifie que l'erreur absolue moyenne a diminué,
  - un gain RMSE positif signifie que les grandes erreurs ont été réduites,
  - un gain de corrélation positif signifie que la forme globale du signal est mieux reproduite.
- **Dérive supprimée** :
  - si elle est positive et significative, la calibration a bien corrigé une composante lente parasite.

Ces résultats sont prometteurs lorsque :
- la corrélation devient très élevée,
- les erreurs diminuent,
- et la dérive résiduelle baisse nettement.

Cependant, une corrélation élevée seule ne suffit pas à conclure que le signal est parfaitement nettoyé :
il faut aussi vérifier simultanément la baisse de la MAE, de la RMSE et de la dérive.
"""
            )

    st.subheader("Analyse synthétique")

    st.markdown(
        f"""
Le modèle **{best_model_name}** est retenu comme meilleur modèle final car il fournit ici la meilleure
combinaison entre qualité de reconstruction et stabilité instrumentale.

On observe en particulier :
- une **corrélation élevée** avec le signal réel,
- une **baisse de la MAE**,
- une **baisse de la RMSE**,
- et une **réduction nette de la dérive**.

Ces résultats sont encourageants et montrent que la calibration améliore effectivement les estimateurs
issus du filtrage.

Cela ne signifie pas que le signal est parfaitement identique au signal réel, mais cela indique que
le pipeline progresse dans la bonne direction :
- le bruit a été réduit au filtrage,
- la dérive a été corrigée à la calibration,
- et les métriques globales évoluent favorablement.
"""
    )

    st.subheader("Conclusion de l'étape")

    st.markdown(
        f"""
Cette étape confirme que le traitement ne s'arrête pas au filtrage.

Même après sélection du meilleur modèle de filtrage, une calibration reste nécessaire pour :
- corriger la dérive lente,
- réduire le biais instrument,
- et améliorer la cohérence avec le signal réel.

Le modèle retenu à l'issue de cette étape est **{best_model_name}**.  
C'est ce modèle calibré qui sera utilisé dans les étapes suivantes de validation et de comparaison.
"""
    )

    with st.expander("Résumé technique de cette étape"):
        st.markdown(
            f"""
- **Entrée de cette étape** : meilleurs signaux filtrés issus de l'étape 3
- **Méthode** : calibration de chaque modèle filtré
- **Paramètre principal** : degré du polynôme de dérive = `{drift_degree}`
- **Option** : préservation de la moyenne = `{preserve_mean}`
- **Modèle final retenu** : `{best_model_name}`
- **Objectif atteint** : sélectionner le meilleur signal après calibration à partir des meilleurs estimateurs filtrés
"""
        )

# =============================================================================
# PARTIE 5
# =============================================================================
elif menu_step == "5️⃣ VALIDATION - Normes ESA/JAXA":
    st.header("5️⃣ Validation selon une plage de température admissible")

    st.markdown("Plage de validation utilisée ici : **-50°C à +100°C**")

    if st.session_state.dbsc_calibrated is None:
        st.warning("Exécute d'abord l'étape 4 de calibration.")
        st.stop()

    if st.session_state.mag_calibrated is None:
        mag_signal = get_spatial_mean(st.session_state.mag_raw)
        mag_time_h = st.session_state.mag_raw["time_hours"].values
        mag_cal = calibrate_filtered_signal(
            signal_1d=mag_signal,
            time_1d=mag_time_h,
            instrument=st.session_state.mag_raw.attrs.get("instrument", "MAG"),
            drift_degree=2,
            preserve_mean=True
        )
        st.session_state.mag_calibrated = inject_calibrated_signal_in_dataset(
            st.session_state.mag_raw,
            mag_cal["calibrated_signal"]
        )

    dbsc_valid = validate_ds(st.session_state.dbsc_calibrated)
    mag_valid = validate_ds(st.session_state.mag_calibrated)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DBSC")
        st.metric("Min", f"{dbsc_valid['min']:.2f} °C")
        st.metric("Max", f"{dbsc_valid['max']:.2f} °C")
        st.metric("Moyenne", f"{dbsc_valid['mean']:.2f} °C")
        st.metric("Écart-type", f"{dbsc_valid['std']:.2f}")
        st.write("Conforme :", "✅" if dbsc_valid["valid_range"] else "❌")

    with col2:
        st.subheader("MAG")
        st.metric("Min", f"{mag_valid['min']:.2f} °C")
        st.metric("Max", f"{mag_valid['max']:.2f} °C")
        st.metric("Moyenne", f"{mag_valid['mean']:.2f} °C")
        st.metric("Écart-type", f"{mag_valid['std']:.2f} °C")
        st.write("Conforme :", "✅" if mag_valid["valid_range"] else "❌")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axvspan(-50, 100, alpha=0.3, color="green", label="Plage autorisée")
    ax.plot([dbsc_valid["min"], dbsc_valid["max"]], [1, 1], "o-", color="orange", linewidth=4, markersize=10, label="DBSC")
    ax.plot([mag_valid["min"], mag_valid["max"]], [0, 0], "o-", color="blue", linewidth=4, markersize=10, label="MAG")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MAG", "DBSC"])
    ax.set_xlabel("Température [°C]")
    ax.set_title("Validation des gammes de température")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# =============================================================================
# PARTIE 6
# =============================================================================
elif menu_step == "6️⃣ COMPARAISON DBSC vs MAG":
    st.header("6️⃣ Comparaison croisée DBSC / MAG")

    if st.session_state.dbsc_calibrated is None:
        st.warning("Exécute d'abord l'étape 4 de calibration.")
        st.stop()

    if st.session_state.mag_calibrated is None:
        mag_signal = get_spatial_mean(st.session_state.mag_raw)
        mag_time_h = st.session_state.mag_raw["time_hours"].values
        mag_cal = calibrate_filtered_signal(
            signal_1d=mag_signal,
            time_1d=mag_time_h,
            instrument=st.session_state.mag_raw.attrs.get("instrument", "MAG"),
            drift_degree=2,
            preserve_mean=True
        )
        st.session_state.mag_calibrated = inject_calibrated_signal_in_dataset(
            st.session_state.mag_raw,
            mag_cal["calibrated_signal"]
        )

    comp = compare_instruments(st.session_state.dbsc_calibrated, st.session_state.mag_calibrated)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{comp['MAE']:.3f}")
    col2.metric("RMSE", f"{comp['RMSE']:.3f}")
    col3.metric("Corrélation", f"{comp['correlation']:.3f}")

    sig1 = get_spatial_mean(st.session_state.dbsc_calibrated)
    sig2 = get_spatial_mean(st.session_state.mag_calibrated)
    time = st.session_state.dbsc_calibrated["time_hours"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(time, sig1, label=f"DBSC ({st.session_state.dbsc_best_calibrated_model_name})", color="orange", linewidth=1.2)
    axes[0].plot(time, sig2, label="MAG", color="blue", alpha=0.8, linewidth=1.2)
    axes[0].set_xlabel("Temps [heures]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Signaux calibrés")
    axes[0].legend()
    axes[0].grid(True)

    mask = ~(np.isnan(sig1) | np.isnan(sig2))
    if np.sum(mask) >= 3:
        min_v = min(np.min(sig1[mask]), np.min(sig2[mask]))
        max_v = max(np.max(sig1[mask]), np.max(sig2[mask]))
        axes[1].scatter(sig1[mask], sig2[mask], s=10, alpha=0.5)
        axes[1].plot([min_v, max_v], [min_v, max_v], "r--", label="Idéal")

    axes[1].set_xlabel("DBSC")
    axes[1].set_ylabel("MAG")
    axes[1].set_title(f"Corrélation = {comp['correlation']:.3f}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    if np.isfinite(comp["correlation"]):
        if comp["correlation"] > 0.9:
            st.success("✅ Instruments très cohérents.")
        elif comp["correlation"] > 0.7:
            st.info("ℹ️ Instruments assez cohérents.")
        else:
            st.warning("⚠️ Cohérence limitée : vérifier calibration ou bruit.")