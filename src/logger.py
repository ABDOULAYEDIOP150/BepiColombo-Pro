"""
Logger configuration for BepiColombo pipeline.

Objectif
--------
- logging propre et lisible
- sortie console + fichier
- niveaux configurables
- réutilisable dans tout le projet
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name="BepiPipeline",
    level=logging.INFO,
    log_file="logs/pipeline.log",
    max_bytes=5_000_000,
    backup_count=3
):
    """
    Configure et retourne un logger.

    Parameters
    ----------
    name : str
        Nom du logger
    level : int
        Niveau de logging (DEBUG, INFO, WARNING, ERROR)
    log_file : str
        Chemin du fichier de log
    max_bytes : int
        Taille max du fichier avant rotation
    backup_count : int
        Nombre de fichiers de backup

    Returns
    -------
    logger : logging.Logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ⚠️ éviter les doublons si déjà initialisé
    if logger.handlers:
        return logger

    # =========================
    # FORMAT LOG
    # =========================
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # =========================
    # CONSOLE HANDLER
    # =========================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # =========================
    # FILE HANDLER (rotation)
    # =========================
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # =========================
    # ADD HANDLERS
    # =========================
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info("Logger initialisé")

    return logger


# ==========================================================
# FONCTIONS UTILES POUR TON PIPELINE
# ==========================================================

def log_pipeline_step(logger, step_name):
    logger.info(f"--- Étape: {step_name} ---")


def log_parameters(logger, **params):
    for key, value in params.items():
        logger.info(f"{key} = {value}")


def log_metrics(logger, metrics_dict, prefix="metrics"):
    for key, value in metrics_dict.items():
        logger.info(f"{prefix}.{key} = {value}")


def log_warning_if_nan(logger, signal, name="signal"):
    import numpy as np
    nan_count = np.sum(np.isnan(signal))
    if nan_count > 0:
        logger.warning(f"{name}: {nan_count} NaN détectés")