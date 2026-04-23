from src.pipeline import run_pipeline
from src.logger import setup_logger

logger = setup_logger()

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting BepiColombo DBSC pipeline")
        results = run_pipeline()
        logger.info("✅ Pipeline finished successfully")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
