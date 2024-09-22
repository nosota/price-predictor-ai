import logging
from datetime import timedelta

import pandas as pd
from tqdm import tqdm
from db import wait_pgsql_connection, query_smsd_orders
from predict import predict_tariff
from models.ai.processing import prepare_prices

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    wait_pgsql_connection()
    logger.info("Выборка исторических данных ...")

    from_date = '2024-07-01'
    df = query_smsd_orders(from_date)

    pr_df = prepare_prices("in/тарифы июнь 2024_1.csv")

    data = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Предсказание цен"):
        try:
            region_code = row['region_code']
            distance = row['distance']
            crop_id = row['crop_id']
            order_date = row['order_date']
            to_address = row['dest_title']
            # имитируем расчет цены заранее за несколько дней
            today_date = order_date - timedelta(days=5)

            predicted_price = predict_tariff(pr_df, region_code, today_date, order_date, crop_id, distance, to_address)
            real_price = float(row['price_gross_per_ton']) // 100

            disassembled_weight = row['disassembled_weight']
            hours_until_disassembled = row['hours_until_disassembled']
            crop_name = row['crop_name']

            data.append([region_code, order_date, distance, crop_name, disassembled_weight, hours_until_disassembled,
                               real_price, predicted_price, predicted_price - real_price])
        except Exception as ex:
            logger.exception(ex)

    results = pd.DataFrame(data, columns=['region_code', 'order_date', 'distance', 'crop_name', 'weight', 'hours_until_disassembled',
                                    'real_price', 'predicted_price', 'inaccuracy'])
    results_sorted = results.sort_values(by='distance')
    results_sorted.to_excel("out/Результаты проверки формулы.xlsx", index=False, engine='openpyxl')  # index=False prevents saving the index
