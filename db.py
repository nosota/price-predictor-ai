import logging
import math
import time

import pandas as pd
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


def query_smsd_orders(from_date: str) -> pd.DataFrame:
    with psycopg2.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stat.delivery_order_id,
                       stat.order_date, 
                       stat.order_id,
                       stat.distance,
                       stat.price_gross_per_ton,
                       stat.price_net_per_ton,
                       stat.disassembled_weight,
                       stat.hours_until_disassembled,
                       stat.region_code,
                       stat.status,
                       stat.ctime,
                       crop_id,
                       crop_name,
                       dest_title
                FROM (SELECT DISTINCT ON (order_id ) order_date,
                                                     order_id,
                                                     delivery_order_id,
                                                     distance,
                                                     price_gross_per_ton,
                                                     price_net_per_ton,
                                                     disassembled_weight,
                                                     hours_until_disassembled,
                                                     region_code,
                                                     status,
                                                     ctime
                      FROM delivery_orders_stat
                      WHERE order_date >= %(from_date)s
                      ORDER BY order_id, ctime, delivery_order_id) AS stat
                         INNER JOIN ((SELECT DISTINCT ON (delivery_order_id) order_id, delivery_order_id, crops_id as crop_id, crops as crop_name
                                      FROM delivery_orders
                                      ORDER BY delivery_order_id) as dorders INNER JOIN (SELECT DISTINCT ON (order_id) order_id, dest_title
                                                                                         FROM orders
                                                                                         WHERE order_date >= %(from_date)s
                                                                                         ORDER BY order_id) as sorders
                                     ON dorders.order_id = sorders.order_id) as t ON stat.delivery_order_id = t.delivery_order_id
            """, {
                "from_date": from_date,
            })

            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
            return df
