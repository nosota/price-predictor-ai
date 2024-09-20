from typing import Tuple, Any

import pandas as pd
import psycopg2
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from init import POSTGRES_DSN


def get_historical_data(num_days, region_code, today_date, crops_id) -> tuple[DataFrame, Any]:

    to_date = today_date + num_days

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
                       crops_id,
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
                      WHERE order_date <= %(to_date)s AND region_code == %(region_code)s
                      ORDER BY order_id, ctime, delivery_order_id) AS stat
                         INNER JOIN ((SELECT DISTINCT ON (delivery_order_id) order_id, delivery_order_id, crops_id, crops as crop_name
                                      FROM delivery_orders WHERE crops_id = %(crops_id)s
                                      ORDER BY delivery_order_id) as dorders INNER JOIN (SELECT DISTINCT ON (order_id) order_id, dest_title
                                                                                         FROM orders
                                                                                         WHERE order_date <= %(to_date)s 
                                                                                         ORDER BY order_id) as sorders
                                     ON dorders.order_id = sorders.order_id) as t ON stat.delivery_order_id = t.delivery_order_id
            """, {
                "to_date": to_date,
                "crops_id": crops_id,
                "region_code": region_code
            })

            rows = cur.fetchall()
            tmp_df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

            le = LabelEncoder()
            tmp_df['dest_id'] = le.fit_transform(tmp_df['dest_title'])

            tmp_df = tmp_df.set_index(['date', 'id'])
            tmp_df.sort_index(inplace=True)

            return tmp_df, le.classes_
