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

        predicted_price = predict_tariff(region_code, order_date, crop_id, distance)

        price_gross_per_ton = float(row['price_gross_per_ton']) // 100
        disassembled_weight = row['disassembled_weight']
        hours_until_disassembled = row['hours_until_disassembled']
        dest_title = row['dest_title']
        crop_name = row['crop_name']
