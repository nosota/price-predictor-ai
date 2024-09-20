import logging
import math
import time

import psycopg2
from dateutil import parser
from psycopg2 import OperationalError

from init import POSTGRES_DSN

logger = logging.getLogger(__name__)


def wait_pgsql_connection():
    while True:
        try:
            logger.info("Service is waiting for connection to PostgreSQL ...")
            with psycopg2.connect(POSTGRES_DSN) as conn:
                pass

            logger.info(f"Service has connected to PostgreSQL.")
            return
        except OperationalError as e:
            logger.error(str(e))
            time.sleep(2)

