import logging
import os
import ssl
import warnings

from dotenv import load_dotenv

load_dotenv(verbose=True)  # .env is used only for DEV env, PROD or Docker doesn't use the file

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.environ['LOG_LEVEL']
)
logging.getLogger().setLevel(os.environ['LOG_LEVEL'])

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# PostgreSQL
POSTGRES_USER = os.environ['POSTGRES_USER']
logger.info(f"POSTGRES_USER = {POSTGRES_USER}")

POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
logger.info(f"POSTGRES_PASSWORD = **********")

POSTGRES_HOST = os.environ['POSTGRES_HOST']
logger.info(f"POSTGRES_HOST = {POSTGRES_HOST}")

POSTGRES_DB = os.environ['POSTGRES_DB']
logger.info(f"POSTGRES_DB = {POSTGRES_DB}")

POSTGRES_PORT = os.environ['POSTGRES_PORT']
logger.info(f"POSTGRES_PORT = {POSTGRES_PORT}")

POSTGRES_DSN=f"dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD} host={POSTGRES_HOST} port={POSTGRES_PORT}"

