import logging
import pandas as pd
from tqdm import tqdm
from db import wait_pgsql_connection, query_smsd_orders
from init import MAX_WORKERS
from predict import predict_tariff
from models.ai.processing import prepare_prices
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def process_row(row, pr_df):
    try:
        region_code = row['region_code']
        distance = row['distance']
        crop_id = row['crop_id']
        order_date = row['order_date']
        to_address = row['dest_title']
        today_date = order_date

        predicted_price = predict_tariff(pr_df, region_code, today_date, order_date, crop_id, distance, to_address)
        real_price = float(row['price_gross_per_ton']) // 100

        disassembled_weight = row['disassembled_weight']
        hours_until_disassembled = row['hours_until_disassembled']
        crop_name = row['crop_name']

        return {
            'region_code': region_code,
            'order_date': order_date,
            'distance': distance,
            'crop_name': crop_name,
            'weight': disassembled_weight,
            'hours_until_disassembled': hours_until_disassembled,
            'real_price': real_price,
            'predicted_price': predicted_price,
            'inaccuracy': predicted_price - real_price
        }
    except Exception as ex:
        logger.exception(ex)
        return None

if __name__ == '__main__':
    wait_pgsql_connection()

    from_date = '2024-07-01'
    df = query_smsd_orders(from_date)

    pr_df = prepare_prices("in/тарифы июнь 2024_1.csv")

    # Создаем пустой DataFrame для результатов
    results = pd.DataFrame(columns=['region_code', 'order_date', 'distance', 'crop_name', 'weight', 'hours_until_disassembled',
                                    'real_price', 'predicted_price', 'inaccuracy'])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row, pr_df): index for index, row in df.iterrows()}

        for future in tqdm(as_completed(futures), total=len(df)):
            result = future.result()
            if result is not None:
                results = results.append(result, ignore_index=True)

    # Сортировка и запись результатов в Excel
    results_sorted = results.sort_values(by='distance')
    results_sorted.to_excel("out/Результаты проверки формулы.xlsx", index=False)
