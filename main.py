from src.pipeline import run_pipeline
from src.logger import setup_logger

logger = setup_logger()

try:
    logger.info("Démarrage pipeline BepiColombo")

    result = run_pipeline()

    logger.info(f"Résultat : {result}")

except Exception as e:
    logger.exception("❌ Erreur dans le pipeline")
    raise