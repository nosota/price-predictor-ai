from typing import Optional
import numpy as np
import pandas as pd
from hist_data import get_historical_data
from my_types import LoadingMethod
from processing import SeriesPredictor, prepare_prices, filter_par_df, get_pars, model_par_df, make_working_df, \
    PrecedentPredictor
from utils import round_to_nearest_10


def get_dest_id(addresses, to_address):
    for index, addr in enumerate(addresses):
        if to_address == addr:
            return index
    return -1


def use_default_model() -> float:
    # TODO: моя модель
    return 0.0


def use_precedent_model(pr_df: pd.DataFrame, historical_data: pd.DataFrame, region_code, order_date, distance,
                        crops_id, dest_id) -> float:
    predictor = PrecedentPredictor(tarif_df = pr_df, eps = 0.1, region_code = region_code)
    predictor.fit(historical_data)

    test_df = pd.DataFrame(columns=['distance', 'date', 'region_code', 'crops_id', 'dest_id'])
    new_row = {'distance': distance, 'date': order_date, 'region_code': region_code, 'crops_id': crops_id, 'dest_id': dest_id}

    # As of pandas 2.0, append (previously deprecated) was removed.
    # test_df = test_df.append(new_row, ignore_index=True)
    #
    # Convert the new_row dictionary to a DataFrame
    new_row_df = pd.DataFrame([new_row])
    # Concatenate the new row DataFrame to the existing DataFrame
    test_df = pd.concat([test_df, new_row_df], ignore_index=True)

    predictor.fit(historical_data)
    res = predictor.predict(test_df)
    predicted_price = res[0]
    return predicted_price


def predict_tariff(pr_df, region_code, today_date, order_date, crop_id, distance, to_address: str) -> float:
    tariff = _predict_tariff_impl(pr_df, region_code, today_date, order_date, crop_id, distance, to_address)
    tariff = tariff // 100
    return round_to_nearest_10(tariff)


def _predict_tariff_impl(pr_df, region_code, today_date, order_date, crop_id, distance, to_address: str) -> float:
    fp = filter_par_df(np.arange(1, 4), np.append(np.arange(2, 11), 0), np.append(np.arange(2, 6), 0))
    mp = model_par_df(['fourier', 'poly'], np.arange(2, 20), np.arange(0, 10), np.append(np.arange(4, 9), 0),
                      np.arange(0, 7))

    if 50 <= distance < 300:
        historical_data, addresses = get_historical_data(11, region_code, today_date, crop_id)  # TODO: переписать на базу, вычитать историю за 11 дней
        if len(historical_data) >= 15:
            filt_pars, mod_pars = get_pars(fp, mp, 67, 23)
            form_type = 'k0k1'  # тип формулы цены
            predictor = SeriesPredictor(tarif_df=pr_df, fpars=filt_pars, mpars=mod_pars,
                                        region_code=region_code, form_type=form_type)
            try:
                predictor.fit(historical_data, date=today_date)
                horizon = order_date - today_date
                res = predictor.predict(horizon, horizon)
                res['target_pred'] = res['target_pred'] / res['distance']  # переводим в тонна-километр
                estimated_price = res['target_pred']
                return estimated_price
            except:
                # Прецедентная модель
                dest_id = get_dest_id(addresses, to_address)  # returns -1 if not found
                return use_precedent_model(pr_df, historical_data, region_code, order_date, distance, crop_id, dest_id)
        else:
            # Прецедентная модель
            dest_id = get_dest_id(addresses, to_address)  # returns -1 if not found
            return use_precedent_model(pr_df, historical_data, region_code, order_date, distance, crop_id, dest_id)

    elif distance >= 300:
        historical_data, addresses = get_historical_data(5, region_code, today_date, crop_id)  # TODO: переписать на базу, вычитать историю за 5 дней
        if len(historical_data) >= 15:
            filt_pars, mod_pars = get_pars(fp, mp, 52, 127)
            form_type = 'k1'  # тип формулы цены
            predictor = SeriesPredictor(tarif_df=pr_df, fpars=filt_pars, mpars=mod_pars,
                                        region_code=region_code, form_type=form_type)
            try:
                predictor.fit(historical_data, date=today_date)
                horizon = order_date - today_date
                res = predictor.predict(horizon, horizon)
                res['target_pred'] = res['target_pred'] / res['distance']  # переводим в тонна-километр
                estimated_price = res['target_pred']
                return estimated_price
            except:
                # Прецедентная модель
                dest_id = get_dest_id(addresses, to_address)  # returns -1 if not found
                return use_precedent_model(pr_df, historical_data, region_code, order_date, distance, crop_id, dest_id)
        else:
            # Прецедентная модель
            dest_id = get_dest_id(addresses, to_address)  # returns -1 if not found
            return use_precedent_model(pr_df, historical_data, region_code, order_date, distance, crop_id, dest_id)
    else:
        # Моя модель
        return use_default_model()
