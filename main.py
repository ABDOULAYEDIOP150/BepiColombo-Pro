"""
Main entry point for the BepiColombo DBSC pipeline.

Rôle
----
Ce script permet d'exécuter le pipeline complet sans interface Streamlit.
Il est utile pour :

- les exécutions locales en ligne de commande,
- les tests rapides,
- les environnements Docker,
- les pipelines CI/CD,
- les logs de production.

Différence avec streamlit_app.py
--------------------------------
- streamlit_app.py : interface interactive de visualisation
- main.py : exécution automatique, non interactive, du pipeline
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any

from src.pipeline import run_pipeline
from src.logger import setup_logger


logger = setup_logger()


def make_json_serializable(obj: Any) -> Any:
    """
    Convertit récursivement un objet Python en structure sérialisable JSON.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_results(results: dict[str, Any], output_dir: str = "outputs") -> Path:
    """
    Sauvegarde les résultats du pipeline au format JSON.

    Parameters
    ----------
    results : dict
        Résultats renvoyés par le pipeline.
    output_dir : str
        Dossier de sortie.

    Returns
    -------
    Path
        Chemin du fichier JSON créé.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"pipeline_results_{timestamp}.json"

    serializable_results = make_json_serializable(results)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    return file_path


def print_summary(results: dict[str, Any]) -> None:
    """
    Affiche un résumé lisible des résultats principaux.
    """
    logger.info("Résumé des résultats du pipeline :")

    reliability_score = results.get("reliability_score")
    if reliability_score is not None:
        logger.info("  - Reliability score : %s", reliability_score)

    filter_metrics = results.get("filter_metrics", {})
    if filter_metrics:
        logger.info("  - Filter metrics : %s", filter_metrics)

    calibration_metrics = results.get("calibration_metrics", {})
    if calibration_metrics:
        logger.info("  - Calibration metrics : %s", calibration_metrics)

    dbsc_validation = results.get("dbsc_validation", {})
    if dbsc_validation:
        logger.info("  - DBSC validation : %s", dbsc_validation)

    mag_validation = results.get("mag_validation", {})
    if mag_validation:
        logger.info("  - MAG validation : %s", mag_validation)

    comparison_metrics = results.get("comparison_metrics", {})
    if comparison_metrics:
        logger.info("  - DBSC vs MAG : %s", comparison_metrics)


def main() -> int:
    """
    Lance le pipeline principal.

    Returns
    -------
    int
        Code de retour shell :
        - 0 si succès
        - 1 si erreur
    """
    logger.info("🚀 Starting BepiColombo DBSC pipeline")

    try:
        results = run_pipeline()

        if not isinstance(results, dict):
            logger.error("Le pipeline a retourné un objet inattendu : %s", type(results))
            return 1

        print_summary(results)

        output_file = save_results(results)
        logger.info("✅ Pipeline finished successfully")
        logger.info("📁 Results saved to: %s", output_file)

        return 0

    except KeyboardInterrupt:
        logger.warning("⏹️ Pipeline interrupted by user")
        return 1

    except Exception as exc:
        logger.error("❌ Pipeline failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())