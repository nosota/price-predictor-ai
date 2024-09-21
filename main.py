from datetime import datetime

import pandas as pd
from tqdm import tqdm
from db import wait_pgsql_connection, query_smsd_orders
from predict import predict_tariff
from processing import prepare_prices

if __name__ == '__main__':
    wait_pgsql_connection()

    from_date = '2024-07-01'
    df = query_smsd_orders(from_date)

    pr_df = prepare_prices("in/тарифы июнь 2024_1.csv")

    results = pd.DataFrame(columns=['region_code', 'order_date', 'distance', 'crop_name', 'weight', 'hours_until_disassembled',
                                    'real_price', 'predicted_price', 'inaccuracy'])

    for index, row in tqdm(df.iterrows(), total=len(df)):
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

        df.loc[len(df)] = [region_code, order_date, distance, crop_name, disassembled_weight, hours_until_disassembled,
                           real_price, predicted_price, predicted_price - real_price]

    results_sorted = results.sort_values(by='distance')
    df.to_excel("out/Результаты проверки формулы.xlsx", index=False)  # index=False prevents saving the index
