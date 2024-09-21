from datetime import datetime

from tqdm import tqdm
from db import wait_pgsql_connection, query_smsd_orders
from predict import predict_tariff


if __name__ == '__main__':
    wait_pgsql_connection()

    from_date = '2024-07-01'
    df = query_smsd_orders(from_date)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        region_code = row['region_code']
        distance = row['distance']
        crop_id = row['crop_id']
        order_date = row['order_date']
        to_address = row['dest_title']
        today_date = order_date

        # today_date = datetime.strptime(today_date, "%Y-%m-%d").date()
        # order_date = datetime.strptime(order_date, "%Y-%m-%d").date()

        predicted_price = predict_tariff(region_code, today_date, order_date, crop_id, distance, to_address)
        predicted_price = predicted_price // 100
        price_gross_per_ton = float(row['price_gross_per_ton']) // 100

        print(f'price_gross_per_ton = {price_gross_per_ton} vs. predicted_price = {predicted_price}')

        disassembled_weight = row['disassembled_weight']
        hours_until_disassembled = row['hours_until_disassembled']
        crop_name = row['crop_name']
