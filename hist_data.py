from datetime import timedelta
from typing import Any
import numpy as np
import pandas as pd
import psycopg2
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from init import POSTGRES_DSN


# Все заявки, которые разбирались дольше указанного в right_price_time времени считать непопулярными
# по причине не верно установленной цены.
def get_historical_data(num_days, region_code, today_date, crops_id, right_price_time=24, target_key='price_gross_per_ton') -> tuple[DataFrame, Any]:
    from_date = today_date - timedelta(days=num_days)

    with psycopg2.connect(POSTGRES_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT stat.delivery_order_id,
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
                  WHERE ctime BETWEEN %(from_date)s AND %(today_date)s
                    AND hours_until_disassembled <= 24
                    AND region_code = %(region_code)s
                  ORDER BY order_id, ctime, delivery_order_id) AS stat
                     INNER JOIN
                 (
                     (SELECT DISTINCT ON (delivery_order_id) order_id, delivery_order_id, crops_id
                      FROM delivery_orders
                      WHERE crops_id = %(crops_id)s
                      ORDER BY delivery_order_id) as dorders
                         INNER JOIN
                         (SELECT DISTINCT ON (order_id) order_id, dest_title
                          FROM orders
                          ORDER BY order_id) as sorders
                     ON dorders.order_id = sorders.order_id
                     ) as t
                 ON stat.delivery_order_id = t.delivery_order_id            
            """, {
                "from_date": from_date,
                "today_date": today_date,
                "crops_id": crops_id,
                "region_code": region_code,
                "right_price_time": right_price_time
            })

            rows = cur.fetchall()
            tmp_df = pd.DataFrame(columns=['date', 'id', 'month', 'distance', 'target', 'targ_price', 'region_code',
                                           'crops_id', 'day_of_week', 'hours', 'dest_id'])
            if len(rows) > 0:
                filtered_df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])

                date_key = 'ctime'  # для правильности построения временного ряда нужен ctime, в не order_date!
                tmp_df['date'] = filtered_df[date_key].dt.floor("D")
                tmp_df['id'] = filtered_df['order_id']
                tmp_df['month'] = filtered_df[date_key].dt.month
                tmp_df['distance'] = filtered_df['distance'].to_numpy()
                tmp_df['target'] = filtered_df[target_key]
                tmp_df['targ_price'] = tmp_df['target'].to_numpy() / tmp_df['distance']
                tmp_df['region_code'] = filtered_df['region_code'].to_numpy()
                tmp_df['crops_id'] = filtered_df['crops_id'].to_numpy()
                tmp_df['day_of_week'] = tmp_df['date'].dt.day_of_week
                tmp_df['hours'] = filtered_df['hours_until_disassembled']
                le = LabelEncoder()
                tmp_df['dest_id'] = le.fit_transform(filtered_df['dest_title'])
                tmp_df = tmp_df.set_index(['date', 'id'])
                tmp_df.sort_index(inplace=True)

                return tmp_df, le.classes_
            else:
                return tmp_df, []
