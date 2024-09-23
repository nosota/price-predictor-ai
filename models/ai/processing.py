from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from numpy import fft
from pmdarima.preprocessing import FourierFeaturizer
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn import metrics as metr
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder

base_dist = 50
long_dist = 500


def adjust_types(df):
    df['ctime'] = pd.to_datetime(df['ctime'])


def prepare_prices(csv_name):
    prices = pd.read_csv(csv_name, sep=';')
    prices.loc[prices['till_date'] == '\\N', 'till_date'] = date.today().strftime('%Y-%m-%d %H:%M:%S')
    pr_df = pd.DataFrame()
    pr_df['region_code'] = prices['region_code']
    pr_df['begin_time'] = pd.to_datetime(prices['since_date'], format='%Y-%m-%d %H:%M:%S')
    pr_df['begin_date'] = pr_df['begin_time'].dt.floor("D")
    pr_df['end_time'] = pd.to_datetime(prices['till_date'], format='%Y-%m-%d %H:%M:%S')
    pr_df['end_date'] = pr_df['end_time'].dt.floor("D")
    pr_df['begin_dist'] = prices['range_start']
    pr_df['end_dist'] = 0
    pr_df.loc[prices['range_end'] != '\\N', 'end_dist'] = prices['range_end'][prices['range_end'] != '\\N'].astype(
        'int64')
    pr_df.loc[prices['range_end'] == '\\N', 'end_dist'] = 5000
    pr_df['price'] = prices['price_amount']
    pr_df = pr_df.set_index(['begin_date', 'begin_dist'], drop=False)
    return pr_df


def intervals_for_region(df, region_code):
    region_mask = df['region_code'] == region_code
    intervals = np.transpose(
        np.array([df.loc[region_mask]['begin_dist'].unique(), df.loc[region_mask]['end_dist'].unique()]))
    intervals_count = intervals.shape[0]
    return intervals


def add_diesel(df, diesel_file, diesel_num):
    diesel = pd.read_excel(diesel_file)
    diesel = diesel.set_index('Дата')
    dr = pd.date_range(start=diesel.index.get_level_values(0).min(), end=diesel.index.get_level_values(0).max())
    diesel = diesel.reindex(dr)
    diesel['Российская Федерация'] = diesel['Российская Федерация'].interpolate(method='linear')
    diesel['Южный\nфедеральный округ'] = diesel['Южный\nфедеральный округ'].interpolate(method='linear')
    df['diesel'] = diesel.loc[df.index.get_level_values(0)].iloc[:, diesel_num].to_numpy()
    return df


def make_working_df(df, *, target_key, add_month, diesel_file, adjust_dist, diesel_num=0):
    tmp_df = pd.DataFrame()
    date_key = 'ctime'
    tmp_df['date'] = df[date_key].dt.floor("D")
    tmp_df['id'] = df['order_id']

    if add_month:
        tmp_df['month'] = df[date_key].dt.month

    tmp_df['distance'] = df['distance'].to_numpy()
    if adjust_dist:
        tmp_df['distance'] = np.maximum(tmp_df['distance'].to_numpy(), base_dist)

    tmp_df['target'] = df[target_key]
    tmp_df['targ_price'] = tmp_df['target'].to_numpy() / tmp_df['distance']
    tmp_df['region_code'] = df['region_code'].to_numpy()
    tmp_df['crops_id'] = df['crops_id'].to_numpy()
    tmp_df['day_of_week'] = tmp_df['date'].dt.day_of_week
    tmp_df['hours'] = df['hours_until_disassembled']
    le = LabelEncoder()
    tmp_df['dest_id'] = le.fit_transform(df['dest_title'])
    tmp_df = tmp_df.set_index(['date', 'id'])
    tmp_df.sort_index(inplace=True)

    if diesel_file is not None:
        tmp_df = add_diesel(tmp_df, diesel_file, diesel_num)

    return tmp_df, le.classes_


def exclude_key(df, key, vlist):
    mask = np.zeros(df.shape[0])
    for v in vlist:
        mask = mask | (df[key] == v)
    res = df.loc[~mask].copy()
    return res


def refine_df(df, t_df):
    res = df[(df['targ_price'] > 100) & df['targ_price'] <= (df['targ_price'].quantile(q=0.99))]
    restricted = list(set(df['region_code'].unique()) - set(t_df['region_code'].unique()))
    res = exclude_key(res, 'region_code', restricted)
    return res


def select_by_key(df, key, value):
    res = df[df[key] == value].copy()
    return res


def select_by_dict(df, key_vals):
    res = df.copy()
    for key in key_vals.keys():
        res = res[res[key] == key_vals[key]].copy()
    return res


def df_region_crop(df, region_code=None, crops_id=None):
    sel = dict()
    if region_code != None:
        sel['region_code'] = region_code
    if crops_id != None:
        sel['crops_id'] = crops_id
    return select_by_dict(df, sel)


def fprice_by_date_region(t_df, date, r_id):
    def price_by_dist(dist):
        t_mask = (t_df['begin_date'] <= date) & (t_df['end_date'] >= date) & (t_df['region_code'] == r_id)
        res = np.zeros(dist.shape[0], )
        ind = 0
        for d in dist:
            d_mask = t_mask & (t_df['begin_dist'] < d) & (t_df['end_dist'] >= d)
            res[ind] = d * t_df.loc[d_mask]['price'].mean()
            ind += 1
        return res

    return price_by_dist


def curve_tarif_k1(t_df, date, r_id):
    def line(dist, k1):
        return k1 * fprice_by_date_region(t_df, date, r_id)(dist)

    return line


def curve_tarif_k0_k1(t_df, date, r_id):
    def line(dist, k0, k1):
        return k0 + k1 * fprice_by_date_region(t_df, date, r_id)(dist)

    return line


def region_regression_series(df, t_df, *, window, min_samples=5, min_dists=3, include_near=False, include_long=True,
                             region_code=23):
    delta = pd.Timedelta(days=window - 1)
    # print(delta)
    start_date = df.index.get_level_values(0).min() + delta
    end_date = df.index.get_level_values(0).max()
    date_range = pd.date_range(start_date, end_date)
    ks = pd.DataFrame(index=date_range)
    for date in date_range:
        loc_range = pd.date_range(date - delta, date)
        sub = df[date - delta:date]
        if (not include_near):
            sub = sub.loc[sub['distance'] > base_dist]
        if (not include_long):
            sub = sub.loc[sub['distance'] <= long_dist]
        if ((sub.shape[0] >= min_samples) & (len(sub['distance'].unique()) >= min_dists)):
            d = sub['distance'].to_numpy()
            p = sub['target'].to_numpy()
            fun = curve_tarif_k1(t_df, date, region_code)
            par, cov = curve_fit(fun, d, p, bounds=([0.5], [2]))
            ks.loc[date, 'k1'] = par[0]
    return ks


def region_regression_series_k0_k1(df, t_df, *, window, min_samples=5, min_dists=3, include_near=False,
                                   include_long=True, region_code=23):
    delta = pd.Timedelta(days=window - 1)
    start_date = df.index.get_level_values(0).min() + delta
    end_date = df.index.get_level_values(0).max()
    date_range = pd.date_range(start_date, end_date)
    ks = pd.DataFrame(index=date_range)
    for date in date_range:
        loc_range = pd.date_range(date - delta, date)
        sub = df[date - delta:date]
        if (not include_near):
            sub = sub.loc[sub['distance'] > base_dist]
        if (not include_long):
            sub = sub.loc[sub['distance'] <= long_dist]
        if ((sub.shape[0] >= min_samples) & (len(sub['distance'].unique()) >= min_dists)):
            d = sub['distance'].to_numpy()
            p = sub['target'].to_numpy()
            fun = curve_tarif_k0_k1(t_df, date, region_code)
            par, cov = curve_fit(fun, d, p, bounds=([-200000, 0.5], [200000, 2]))
            ks.loc[date, 'k0'] = par[0]
            ks.loc[date, 'k1'] = par[1]
    return ks


def filter_key(origin):
    return 'f' + origin


def make_filter_keys(origins):
    res = []
    for origin in origins:
        res.append(filter_key(origin))
    return res


def filter(df, use_filter, filter_window, filter_order):
    fdf = pd.DataFrame(index=df.index)
    for key in df.columns:
        if use_filter == True:
            fdf[filter_key(key)] = savgol_filter(df[key].interpolate().to_numpy(), window_length=filter_window,
                                                 polyorder=filter_order)
        else:
            fdf[filter_key(key)] = df[key].interpolate().to_numpy()
    return fdf


def gen_features(x, deg):
    res = np.ones([x.size, deg])
    res[:, 0] = x
    for d in range(1, deg):
        res[:, d] = res[:, d - 1] * x
    return res


def extract_trend(x):
    n = x.shape[0]
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)
    x_notrend = x - p[0] * t
    return x_notrend, p[0]


def predict_spectrum(x, days, n_harm):
    n = x.shape[0]
    detr_x, k = extract_trend(x)
    x_freqdom = fft.fft(detr_x)
    f = fft.fftfreq(n)
    indexes = list(range(n))
    t = np.arange(0, n + days)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    pred = restored_sig + k * t
    return pred[n + days - 1], pred


def predict_arima(x, days, n_harm, period):
    ff = FourierFeaturizer(period, n_harm)
    if n_harm > 0:
        x, X = ff.fit_transform(x)
        x, Xpr = ff.transform(x, n_periods=days)
        m = pm.auto_arima(x, X, error_action='ignore', seasonal=False)
        pred = m.predict(n_periods=days, X=Xpr).to_numpy()
    else:
        m = pm.auto_arima(x, error_action='ignore', seasonal=False)
        pred = m.predict(n_periods=days)
    return pred[-1], pred


def predict_poly(y, days, deg):
    m = ElasticNet()
    x = np.arange(y.shape[0])
    X = gen_features(x, deg)
    m.fit(X, y)
    xtest = np.arange(y.shape[0], y.shape[0] + days)
    Xtest = gen_features(xtest, deg)
    pred = m.predict(Xtest)
    return pred[-1], pred


def predict_by_tarif_and_k(df, t_df, r_id, k_df, keys, key_pred):
    key_k = keys[0]
    res = df.copy()
    res[key_pred] = None
    for date, items in k_df.groupby(level=0):
        # print(items)
        k = items[key_k].to_numpy()[0]
        fun = curve_tarif_k1(t_df, date, r_id)
        if date in df.index.get_level_values(0):
            d = df.loc[date, 'distance'].to_numpy()
            res.loc[date, key_pred] = fun(d, k)
    return res


def predict_by_tarif_and_k0_k1(df, t_df, r_id, k_df, keys, key_pred):
    key_k0 = keys[0]
    key_k1 = keys[1]
    res = df.copy()
    res[key_pred] = None
    for date, items in k_df.groupby(level=0):
        # print(items)
        k0 = items[key_k0].to_numpy()[0]
        k1 = items[key_k1].to_numpy()[0]
        fun = curve_tarif_k0_k1(t_df, date, r_id)
        if date in df.index.get_level_values(0):
            d = df.loc[date, 'distance'].to_numpy()
            res.loc[date, key_pred] = fun(d, k0, k1)
    return res


def predict_by_tarif(df, t_df, r_id, start_date, key_pred):
    res = df.copy()
    res[key_pred] = None
    for date, items in df.loc[start_date:].groupby(level=0):
        fun = fprice_by_date_region(t_df, date, r_id)
        if date in df.index.get_level_values(0):
            d = items['distance'].to_numpy()
            res.loc[date, key_pred] = fun(d)
    return res


def pred_key(origin, d):
    return origin + '_' + 'pred' + str(d)


def make_pred_keys(origins, pred_start, pred_end):
    res = []
    for origin in origins:
        for d in range(pred_start, pred_end + 1):
            res.append(pred_key(origin, d))
    return res


def predict_series(y, *, days, var='fourier', n_harm=0, period=None, deg=4):
    if var == 'fourier':
        px, rx = predict_spectrum(y, days, n_harm)
    if var == 'arima':
        px, rx = predict_arima(y, days, n_harm, period)
    if var == 'poly':
        px, rx = predict_poly(y, days, deg)
    return px, rx


def predict_series_on_date(kdf, pdf, date, *, delta, pred_start=3, pred_end=5, var='fourier', n_harm=0, period=None,
                           deg=4):
    columns = kdf.columns
    end_date = kdf.index.get_level_values(0).max()
    loc_start = date - delta
    loc_end = date
    loc_range = pd.date_range(loc_start, loc_end)
    for key in columns:
        px, rx = predict_series(kdf.loc[pd.date_range(loc_start, loc_end), key].to_numpy(), days=pred_end,
                                n_harm=n_harm, \
                                period=period, deg=deg, var=var)
        # print(rx)
        for i in range(0, pred_end - pred_start + 1):
            if (loc_end + pd.Timedelta(days=pred_end - i)) <= end_date:
                pdf.loc[loc_end + pd.Timedelta(days=pred_end - i), pred_key(key, pred_end - i)] = rx[-1 - i]


def predict_series_once(kdf, *, tr_days, pred_start=3, pred_end=5, var='fourier', n_harm=0, period=None, deg=4):
    delta = pd.Timedelta(days=tr_days - 1)
    columns = kdf.columns
    date = kdf.index.get_level_values(0).max()
    pr_delta_start = pd.Timedelta(days=pred_start)
    pr_delta_end = pd.Timedelta(days=pred_end)
    loc_start = date - delta
    loc_end = date
    loc_range = pd.date_range(loc_start, loc_end)
    pred_range = pd.date_range(date + pr_delta_start, date + pr_delta_end)
    preds = pd.DataFrame(index=pred_range, columns=kdf.columns)
    for key in columns:
        px, rx = predict_series(kdf.loc[pd.date_range(loc_start, loc_end), key].to_numpy(), days=pred_end,
                                n_harm=n_harm, \
                                period=period, deg=deg, var=var)
        for i in range(0, pred_end - pred_start + 1):
            preds.loc[loc_end + pd.Timedelta(days=pred_end - i), key] = rx[-1 - i]
    return preds


def moving_pred(df, tr_days, *, pred_start=3, pred_end=5, n_harm=0, period=None, deg=4, var='fourier'):
    delta = pd.Timedelta(days=tr_days - 1)
    test_delta = pd.Timedelta(days=pred_start)
    start_date = df.index.get_level_values(0).min()
    end_date = df.index.get_level_values(0).max()
    date_range = pd.date_range(start_date + delta, end_date - test_delta)
    preds = pd.DataFrame(index=date_range, columns=make_pred_keys(df.columns, pred_start, pred_end))
    for date in date_range:
        predict_series_on_date(df, preds, date, delta=delta, pred_start=pred_start, pred_end=pred_end, var=var,
                               n_harm=n_harm, period=period, deg=deg)
    return preds


def adjust_quantile(errors, target):
    q = 1
    while errors.quantile(q=q) > target:
        q = q - 0.01
    return errors.quantile(q=q), q


def scores_old(true_Y, pred, d, idx, q, adjust_q, targ, main_metr):
    y1 = true_Y
    y2 = pred
    E = pd.DataFrame(data=(pred - true_Y).to_numpy(), columns=['E'], index=idx, copy=True)
    E['target'] = true_Y
    E['pred'] = pred
    E['AE'] = np.abs(E['E'])
    E['distance'] = d
    minE = E['E'].min()
    maxE = E['E'].max()
    meanE = E['E'].mean()
    mae = metr.mean_absolute_error(y1, y2)
    maxE = metr.max_error(y1, y2)
    rmse = metr.root_mean_squared_error(y1, y2)
    if adjust_q == False:
        adq = q
        qua = E['AE'].quantile(q=adq)
    else:
        qua, adq = adjust_quantile(E['AE'], targ)
    sc_dict = {'minE': minE, 'maxE': maxE, 'meanE': meanE, 'mae': mae, 'maxE': maxE, 'rmse': rmse,
               'quantile_' + str(round(adq * 100, 0)): qua}
    if main_metr == 'None':
        return E, sc_dict
    else:
        return E, sc_dict[main_metr]


def scores(true_Y, pred, d, idx):
    y1 = true_Y
    y2 = pred
    data = (pred - true_Y).to_numpy()
    E = pd.DataFrame(data=data, dtype=float, columns=['errors'], index=idx, copy=True)
    E['distance'] = d
    return E, E['errors'].describe()


def scores_from_df(df, targ_col, pred_col, begin_dist, end_dist):
    mask = (df['distance'] > begin_dist) & (df['distance'] <= end_dist)
    masked = df.loc[mask, ['distance', targ_col, pred_col]]
    sub = masked[masked[pred_col].to_numpy() != None].dropna()
    return scores(sub[targ_col], sub[pred_col], sub['distance'], sub.index)


def scores_from_df_old(df, targ_col, pred_col, begin_dist, end_dist, *, q=0.9, adjust_q=False, targ=100,
                       main_metr='None'):
    mask = (df['distance'] > begin_dist) & (df['distance'] <= end_dist)
    masked = df.loc[mask, ['distance', targ_col, pred_col]]
    sub = masked[masked[pred_col].to_numpy() != None].dropna()
    return scores_old(sub[targ_col], sub[pred_col], sub['distance'], sub.index, q, adjust_q, targ, main_metr)


def disp_errors_old(err_df, postfix=''):
    fig, ax = plt.subplots()
    ax.scatter(err_df['target'], err_df['pred'])
    plt.title('Prediction vs TrueY' + ' ' + postfix)
    plt.ylabel('Prediction')
    plt.xlabel('TrueY')
    ax.figure.set_figwidth(10)
    fig, ax = plt.subplots()
    ax.scatter(err_df['distance'], err_df['E'])
    plt.title('Error vs Distance' + ' ' + postfix)
    plt.ylabel('Error')
    plt.xlabel('Distance')
    ax.figure.set_figwidth(10)
    fig, ax = plt.subplots()
    ax.scatter(err_df.index.get_level_values(0), err_df['E'])
    plt.title('Error vs Date' + ' ' + postfix)
    plt.ylabel('Error')
    plt.xlabel('Date')
    ax.figure.set_figwidth(10)


def disp_errors(err_df, postfix=''):
    fig, ax = plt.subplots()
    err_df['errors'].hist(ax=ax)
    plt.title('Histogram for error ' + postfix)
    fig, ax = plt.subplots()
    err_df.boxplot(column='errors', ax=ax)
    plt.title('errors ' + postfix)


def make_mean_series(df, keys, categories):
    cat_range = range(0, categories.shape[0])
    limits = categories[:, 1]
    date_range = pd.date_range(df.index.get_level_values(0).min(), df.index.get_level_values(0).max())
    midx = pd.MultiIndex.from_product([date_range, limits], names=['date', 'cat'])
    res = pd.DataFrame(index=midx, columns=keys)
    res['distance'] = None
    for date, items in df.groupby(level=0):
        for i in cat_range:
            left = categories[i, 0]
            right = categories[i, 1] if i < categories.shape[0] - 1 else np.inf
            cat_mask = (items['distance'] > left) & (items['distance'] <= right)
            res.loc[(date, limits[i]), 'distance'] = right
            for key in keys:
                res.loc[(date, limits[i]), key] = items.loc[cat_mask][key].mean()
    return res


def show_predictions(k_df, fk_df, preds, hors):
    for key in k_df.columns:
        fig, ax = plt.subplots()
        start_pred = preds[pred_key(filter_key(key), hors[0])].dropna().index.get_level_values(0).min()
        ax.scatter(k_df.loc[start_pred:].index.get_level_values(0), k_df.loc[start_pred:][key])
        ax.figure.set_figwidth(10)
        ax.plot(fk_df.loc[start_pred:][filter_key(key)], label='Filtered coefficient to predict')
        for h in hors:
            ax.plot(preds[pred_key(filter_key(key), h)], label='Prediction for ' + str(h) + ' days')
        ax.legend()
        ax.set_title("Results for " + key)


def targ_score(df, targ_key, *, targ_hor=3, begin_dist=base_dist, end_dist=5000, targ_metr='quantile_90.0'):
    pr_key = pred_key(targ_key, targ_hor)
    return scores_from_df_old(df, targ_key, pr_key, begin_dist, end_dist, q=0.9, adjust_q=False, targ=100,
                              main_metr=targ_metr)


def test_score_df_old(df, targ_key, pred_key, postfix, *, plot_results=True, begin_dist=base_dist, end_dist=5000, q=0.9,
                      adjust_q=True, sep=300, targ=[5000, 10000]):
    if sep == 0:
        print('Results for distances ' + str(begin_dist) + '-' + str(end_dist) + ' km')
    else:
        print('Results for distances < ' + str(sep) + ' km')
        _, sc = scores_from_df_old(df, targ_key, pred_key, begin_dist, sep, q=q, adjust_q=adjust_q, targ=targ[0],
                                   main_metr='None')
        print_scores_old(sc)
        print('Results for distances > ' + str(sep) + ' km')
        _, sc = scores_from_df_old(df, targ_key, pred_key, sep, end_dist, q=q, adjust_q=adjust_q, targ=targ[1],
                                   main_metr='None')
        print_scores_old(sc)
    if plot_results:
        err, _ = scores_from_df_old(df, targ_key, pred_key, begin_dist, end_dist, q=q, adjust_q=False, targ=targ[0],
                                    main_metr='None')
        disp_errors(err, postfix)


def test_score_df(df, targ_key, pred_key, postfix, *, plot_results=True, begin_dist=base_dist, end_dist=5000, sep=300):
    if sep == 0:
        print('Results for distances ' + str(begin_dist) + '-' + str(end_dist) + ' km')
        err, sc = scores_from_df(df, targ_key, pred_key, begin_dist, end_dist)
        print_scores(sc)
        if plot_results:
            disp_errors(err, postfix)
    else:
        print('Results for distances < ' + str(sep) + ' km')
        err, sc = scores_from_df(df, targ_key, pred_key, begin_dist, sep)
        print_scores(sc)
        if plot_results:
            disp_errors(err, postfix)
        print('Results for distances > ' + str(sep) + ' km')
        err, sc = scores_from_df(df, targ_key, pred_key, sep, end_dist)
        print_scores(sc)
        if plot_results:
            disp_errors(err, postfix)


def print_scores_old(sc):
    print('')
    for k in sc.keys():
        print(k, sc[k])
    print('')


def interp_describe(desc):
    M = round(desc['mean'] / 100, 0)
    sig = round(desc['std'] / 100, 0)
    print('Можем предположить, что ошибка распределена нормально с матожиданием в районе ' + str(M) + 'р')
    print('И СКО примерно ' + str(round(desc['std'] / 100, 0)) + 'р')
    print('Тогда существующая ошибка лежит в интервале ' + str(round(M - 2 * sig, 0)) + '..' + str(
        round(M + 2 * sig, 0)) + 'р')


def print_scores(sc):
    print('')
    print(sc)
    interp_describe(sc)
    print('')


def test_score_h_old(df, targ_key, hor, *, plot_results=True, begin_dist=base_dist, end_dist=5000,
                     mean_df=pd.DataFrame(), q=0.9, adjust_q=True, sep=300, targ=[5000, 10000]):
    pr_key = pred_key(targ_key, hor)
    print('Results for target')
    postfix = 'for target on ' + str(hor) + '-days prediction'
    test_score_df_old(df, targ_key, pr_key, postfix, plot_results=plot_results, begin_dist=begin_dist,
                      end_dist=end_dist, q=q, adjust_q=adjust_q, sep=sep, targ=targ)
    if not mean_df.empty:
        print('Results for mean target')
        postfix = 'for mean target on ' + str(hor) + '-days prediction'
        test_score_df_old(mean_df, targ_key, pr_key, postfix, plot_results=plot_results, begin_dist=begin_dist,
                          end_dist=end_dist, q=q, adjust_q=adjust_q, sep=sep, targ=targ)


def test_score_h(df, targ_key, hor, *, plot_results=True, begin_dist=base_dist, end_dist=5000, mean_df=pd.DataFrame(),
                 sep=300):
    pr_key = pred_key(targ_key, hor)
    print('Results for target')
    postfix = 'for target on ' + str(hor) + '-days prediction'
    test_score_df(df, targ_key, pr_key, postfix, plot_results=plot_results, begin_dist=begin_dist, end_dist=end_dist,
                  sep=sep)
    if not mean_df.empty:
        print('Results for mean target')
        postfix = 'for mean target on ' + str(hor) + '-days prediction'
        test_score_df(mean_df, targ_key, pr_key, postfix, plot_results=plot_results, begin_dist=begin_dist,
                      end_dist=end_dist, sep=sep)


def test_score_old(df, targ_key, region_code, t_df, *, plot_results=True, begin_dist=base_dist, end_dist=5000,
                   pred_start=3, pred_end=5, calc_mean=True, q=0.9, adjust_q=True, sep=300, targ=[5000, 10000]):
    if calc_mean:
        pr_keys = make_pred_keys([targ_key], pred_start, pred_end)
        mdf = make_mean_series(df, [targ_key] + pr_keys, intervals_for_region(t_df, region_code))
    else:
        mdf = pd.DataFrame()
    for h in range(pred_start, pred_end + 1):
        print('Results for ' + str(h) + '-day prediction')
        test_score_h_old(df, targ_key, h, plot_results=plot_results, begin_dist=begin_dist, end_dist=end_dist,
                         mean_df=mdf, q=q, \
                         adjust_q=adjust_quantile, sep=sep, targ=targ)


def test_score(df, targ_key, region_code, t_df, *, plot_results=True, calc_mean=True, begin_dist=base_dist,
               end_dist=5000, \
               pred_start=3, pred_end=5, sep=300):
    if calc_mean:
        pr_keys = make_pred_keys([targ_key], pred_start, pred_end)
        mdf = make_mean_series(df, [targ_key] + pr_keys, intervals_for_region(t_df, region_code))
    else:
        mdf = pd.DataFrame()
    for h in range(pred_start, pred_end + 1):
        print('Results for ' + str(h) + '-day prediction')
        test_score_h(df, targ_key, h, plot_results=plot_results, begin_dist=begin_dist, end_dist=end_dist, mean_df=mdf,
                     sep=sep)


def validate_filter_pars(pars):
    u, w, fw, fo = tuple(pars)
    return (w > 0) & (((not u) & (fo == 0) & (fw == 0)) | (u & (fo > 0) & (fw > fo)))


def filter_par_df(w_sizes, fw_sizes, forders):
    u = [True, False]
    index = pd.MultiIndex.from_product([u, w_sizes, fw_sizes, forders],
                                       names=['use_filter', 'window', 'filter_window', 'filter_order'])
    res = pd.DataFrame(index=index).reset_index()
    mask = res.apply(validate_filter_pars, axis=1).astype(bool)
    res = res.loc[mask].reset_index(drop=True)
    return res


def validate_model_pars(pars):
    v, tw, h, p, po = tuple(pars)
    poly_val = (v == 'poly') & (po > 0) & (tw > po) & (tw <= po + 4) & (h == 0) & (p == 0)
    fourier_val = (v == 'fourier') & (tw > h) & (po == 0) & (p == 0) & (h > 0)
    arima_val = (v == 'arima') & (po == 0) & (tw >= 7) & (
                (h > 0) & (h <= 4) & (p >= 2 * h) & (tw >= 2 * p) | (h == 0) & (p == 0))
    return poly_val | fourier_val | arima_val


def model_par_df(vars, train_windows, harms, periods, poly_orders):
    index = pd.MultiIndex.from_product([vars, train_windows, harms, periods, poly_orders],
                                       names=['variant', 'train_window', \
                                              'n_harm', 'period', 'poly_order'])
    res = pd.DataFrame(index=index).reset_index()
    mask = res.apply(validate_model_pars, axis=1).astype(bool)
    res = res.loc[mask].reset_index(drop=True)
    return res


def validate_all_pars(fpars, mpars):
    _, _, fw, _ = tuple(fpars)
    _, tw, _, _, _ = tuple(mpars)
    return (tw >= fw) & validate_filter_pars(fpars) & validate_model_pars(mpars)


def get_pars(fpars, mpars, ind_f, ind_m):
    return fpars.loc[ind_f], mpars.loc[ind_m]


def moving_prediction(fk_df, k_df, *, show_results=True, pred_start=3, pred_end=5, var, train_window, n_harm, period,
                      poly_order):
    preds = moving_pred(fk_df, train_window, pred_start=3, pred_end=5, n_harm=n_harm, period=period, deg=poly_order,
                        var=var)
    merged = k_df.merge(fk_df, left_index=True, right_index=True)
    if show_results:
        if pred_start == pred_end:
            hors = [pred_start]
        else:
            hors = [pred_start, pred_end]
        show_predictions(k_df, fk_df, preds, hors)
    return preds, merged


def restore_prediction(df, pred_df, t_df, region_code, restore_fun, keys, pred_start=3, pred_end=5):
    res = restore_fun(df, t_df, region_code, pred_df, make_pred_keys(make_filter_keys(keys), pred_start, pred_start),
                      pred_key('target', pred_start))
    for h in range(pred_start + 1, pred_end + 1):
        res = restore_fun(res, t_df, region_code, pred_df, make_pred_keys(make_filter_keys(keys), h, h),
                          pred_key('target', h))
    return res


def estimate_model(df, fdf, kdf, t_df, mpars, restore_fun, *, region_code=23, targ_hor=3, targ_metr='quantile_90.0',
                   begin_dist=base_dist, end_dist=5000):
    var, train_window, n_harm, period, poly_order = mpars
    pred_df, fdf = moving_prediction(fdf, kdf, show_results=False, train_window=train_window, pred_start=targ_hor,
                                     pred_end=targ_hor, n_harm=n_harm, period=period, poly_order=poly_order, var=var)
    res_df = restore_prediction(df, pred_df, t_df, region_code, restore_fun, kdf.columns, pred_start=targ_hor,
                                pred_end=targ_hor)
    _, sc = targ_score(res_df, 'target', targ_hor=targ_hor, begin_dist=base_dist, end_dist=end_dist,
                       targ_metr='quantile_90.0')
    return sc


def test_filter_model(df, t_df, fpars, mpars, reg_fun, restore_fun, *, region_code=23, crops_id=None, pred_start=3,
                      pred_end=5, sep=300):
    if not validate_all_pars(fpars, mpars):
        return pd.DataFrame()
    min_samples = 2
    min_dists = 2
    use_filter, window, filter_window, filter_order = fpars
    var, train_window, n_harm, period, poly_order = mpars
    sel = df_region_crop(df, region_code, crops_id)
    kdf = reg_fun(sel, t_df, window=window, min_samples=min_samples, min_dists=min_dists, include_near=False,
                  include_long=True, \
                  region_code=region_code)
    fdf = filter(kdf, use_filter, filter_window, filter_order)
    pred_df, fdf = moving_prediction(fdf, kdf, show_results=True, train_window=train_window, \
                                     pred_start=pred_start, pred_end=pred_end, n_harm=n_harm, period=period,
                                     poly_order=poly_order, var=var)
    res_df = restore_prediction(sel, pred_df, t_df, region_code, restore_fun, kdf.columns, pred_start=pred_start,
                                pred_end=pred_end)
    test_score(res_df, 'target', region_code, t_df, plot_results=True, pred_start=pred_start, pred_end=pred_end,
               sep=sep)
    return res_df


def optimize_filter_model(df, t_df, fp_df, mp_df, reg_fun, restore_fun, *, region_code=23, crops_id=None, targ_hor=3,
                          targ_metr='quantile_90.0'):
    best = {'value': np.inf}
    fpar_cnt = fp_df.shape[0]
    sel = df_region_crop(df, region_code, crops_id)
    for item in fp_df.iterrows():
        ind_f, parf = item
        use_filter, window, filter_window, filter_order = tuple(parf)
        kdf = reg_fun(sel, t_df, window=window, min_samples=2, min_dists=2, include_near=False, include_long=True,
                      region_code=region_code)
        fdf = filter(kdf, use_filter, filter_window, filter_order)
        mpar_cnt = mp_df.shape[0]
        for mitem in mp_df.iterrows():
            ind_m, parm = mitem
            # print('model params '+str(ind_m+1)+'/'+str(mpar_cnt))
            if validate_all_pars(parf, parm):
                sc = estimate_model(sel, fdf, kdf, t_df, tuple(parm), restore_fun, region_code=region_code,
                                    targ_hor=targ_hor, \
                                    targ_metr=targ_metr, begin_dist=base_dist, end_dist=5000)
                if best['value'] > sc:
                    best['value'] = sc
                    best['filter_p'] = parf
                    best['model_p'] = parm
                    print('best updated: ', sc, ' filter params ' + str(ind_f + 1) + '/' + str(fpar_cnt),
                          ' model params ' + str(ind_m + 1) + '/' + str(mpar_cnt))
    return best


# Предсказание по тарифу
def predict_4dist_by_tarif(dist, date, r_id, t_df):
    max_left = t_df.index.get_level_values(0).max()
    t_mask = (t_df['begin_date'].dt.date <= date) & (t_df['end_date'].dt.date >= date) & (t_df['region_code'] == r_id)
    t_mask = t_mask&(t_df['begin_dist']<dist)&(t_df['end_dist']>=dist)
    return dist*t_df.loc[t_mask]['price'].mean()


# Предсказание по расстоянию, пункту назначения и дню недели (в порядке приоритета)
# здесь в параметрах тоже есть культура, но она не используется, добавлена для совместимости разных версий функций
def predict_by_precedent_dow(df, dist, date, crops_id, region_code, dest_id, tarif, dist_eps=0.1):
    day = date.day_of_week
    # смотрим, были ли вообще перевозки на это расстояние
    dist_mask = (df['distance'] == dist)
    if any(dist_mask):
        mask = dist_mask & (df['day_of_week'] == day) & (df['dest_id'] == dest_id)
        # если был точно такой заказ - возвращаем его
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_e'
        # если не было, пробуем с другим днём недели
        mask = dist_mask & (df['dest_id'] == dest_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_d_e'
        # если и тут мимо, пробуем с другим пунктом, но этим днём недели
        mask = dist_mask & (df['day_of_week'] == day)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_d'
        # если не вышло найти пункт и день недели, просто возвращаем свежую перевозку
        # на это расстояние
        return df.loc[dist_mask].tail(1)['target'].to_numpy()[0], 'e_d_d'
    # если не было перевозок с точным совпадением, смотрим окрестность
    dist_mask = (df['distance'] >= dist * (1 - dist_eps)) & (df['distance'] <= dist * (1 + dist_eps))
    if any(dist_mask):
        sub = df.loc[dist_mask].copy()
        # считаем дельту и сортируем в обратном порядке
        sub['delta'] = np.abs(sub['distance'].to_numpy() - dist)
        sub.sort_values(["delta"], ascending=False)
        # если был похожий заказ - возвращаем его
        mask = (sub['day_of_week'] == day) & (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_e'
        # если не было, пробуем с другим днём недели
        mask = sub['dest_id'] == dest_id
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_d_e'
        # если и тут мимо, пробуем с другим пунктом, но этим днём недели
        mask = sub['day_of_week'] == day
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_d'
        # если не вышло найти пункт и день недели, просто возвращаем ближайшую свежую перевозку
        # в этой окрестности
        sub.sort_values(["delta", "date"], ascending=[False, True])
        p = sub.tail(1)['targ_price'].to_numpy()[0]
        return p * dist, 'd_d_d'
    # по умолчанию, берём тариф
    return predict_4dist_by_tarif(dist, date, region_code, tarif), 't'


# Предсказание по расстоянию, пункту назначения и культуре (в порядке приоритета)
def predict_by_precedent_crops(df, dist, date, crops_id, region_code, dest_id, tarif, dist_eps=0.1):
    # смотрим, были ли вообще перевозки на это расстояние
    dist_mask = (df['distance'] == dist)
    if any(dist_mask):
        mask = dist_mask & (df['crops_id'] == crops_id) & (df['dest_id'] == dest_id)
        # если был точно такой заказ - возвращаем его
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_e'
        # если не было, пробуем с другим пунктом, но этой культурой
        mask = dist_mask & (df['crops_id'] == crops_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_d'
        # если и тут мимо,пробуем с другой культурой
        mask = dist_mask & (df['dest_id'] == dest_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_d_e'
        # если не вышло найти пункт и день недели, просто возвращаем свежую перевозку
        # на это расстояние
        return df.loc[dist_mask].tail(1)['target'].to_numpy()[0], 'e_d_d'
    # если не было перевозок с точным совпадением, смотрим окрестность
    dist_mask = (df['distance'] >= dist * (1 - dist_eps)) & (df['distance'] <= dist * (1 + dist_eps))
    if any(dist_mask):
        sub = df.loc[dist_mask].copy()
        # считаем дельту и сортируем в обратном порядке
        sub['delta'] = np.abs(sub['distance'].to_numpy() - dist)
        sub.sort_values(["delta"], ascending=False)
        # если был похожий заказ - возвращаем его
        mask = (sub['crops_id'] == crops_id) & (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_e'
        # если не было, пробуем с другим пунктом
        mask = sub['crops_id'] == crops_id
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_d'
            # если и тут мимо, пробуем с другой культурой, но этим пунктом
        mask = sub['dest_id'] == dest_id
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_d_e'
        # если не вышло найти пункт и культуру, просто возвращаем ближайшую свежую перевозку
        # в этой окрестности
        sub.sort_values(["delta", "date"], ascending=[False, True])
        p = sub.tail(1)['targ_price'].to_numpy()[0]
        return p * dist, 'd_d_d'
    # по умолчанию, берём тариф
    return predict_4dist_by_tarif(dist, date, region_code, tarif), 't'
    # Предсказание по расстоянию, пункту назначения, культуре и дню недели (в порядке приоритета)


def predict_by_precedent_crops_dow(df, dist, date, crops_id, region_code, dest_id, tarif, dist_eps=0.1):
    day = date.day_of_week
    # смотрим, были ли вообще перевозки на это расстояние
    dist_mask = (df['distance'] == dist)
    if any(dist_mask):
        mask = dist_mask & (df['crops_id'] == crops_id) & (df['day_of_week'] == day) & (df['dest_id'] == dest_id)
        # если был точно такой заказ - возвращаем его
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_e_e'
        # если не было, пробуем с другим днём недели но этими культурой и пунктом
        mask = dist_mask & (df['crops_id'] == crops_id) & (df['dest_id'] == dest_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_d_e'
        # если тут мимо,пробуем с другим пунктом, но этими культурой и днём недели
        mask = dist_mask & (df['crops_id'] == crops_id) & (df['day_of_week'] == day)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_e_d'
        # если снова мимо,пробуем с другим днём недели и пунктом
        mask = dist_mask & (df['crops_id'] == crops_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_e_d_d'
        # если не вышло найти культуру,пробуем с другой культурой, но этими пунктом и днём недели
        mask = dist_mask & (df['day_of_week'] == day) & (df['dest_id'] == dest_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_d_e_e'
        # если тут мимо,пробуем с другим днём недели
        mask = dist_mask & (df['dest_id'] == dest_id)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_d_d_e'
        # если тут мимо,пробуем с другим пунктом
        mask = dist_mask & (df['day_of_week'] == day)
        if any(mask):
            return df.loc[mask].tail(1)['target'].to_numpy()[0], 'e_d_e_d'
        # если не вышло найти культуру пункт и день недели, просто возвращаем свежую перевозку
        # на это расстояние
        return df.loc[dist_mask].tail(1)['target'].to_numpy()[0], 'e_d_d_d'
    # если не было перевозок с точным совпадением, смотрим окрестность
    dist_mask = (df['distance'] >= dist * (1 - dist_eps)) & (df['distance'] <= dist * (1 + dist_eps))
    if any(dist_mask):
        sub = df.loc[dist_mask].copy()
        # считаем дельту и сортируем в обратном порядке
        sub['delta'] = np.abs(sub['distance'].to_numpy() - dist)
        sub.sort_values(["delta"], ascending=False)
        # если был похожий заказ - возвращаем его
        mask = (sub['crops_id'] == crops_id) & (sub['day_of_week'] == day) & (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_e_e'
        # если не было, пробуем с другим днём недели но этими культурой и пунктом
        mask = (sub['crops_id'] == crops_id) & (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_d_e'
            # если тут мимо,пробуем с другим пунктом, но этими культурой и днём недели
        mask = (sub['crops_id'] == crops_id) & (sub['day_of_week'] == day)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_e_d'
        # если снова мимо,пробуем с другим днём недели и пунктом
        mask = (sub['crops_id'] == crops_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_e_d_d'
        # если не вышло найти культуру,пробуем с другой культурой, но этими пунктом и днём недели
        (sub['day_of_week'] == day) & (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_d_e_e'
        # если тут мимо,пробуем с другим днём недели
        mask = (sub['dest_id'] == dest_id)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_d_d_e'
        # если тут мимо,пробуем с другим пунктом
        mask = (sub['day_of_week'] == day)
        if any(mask):
            p = sub.loc[mask].tail(1)['targ_price'].to_numpy()[0]
            return p * dist, 'd_d_e_d'
        # если не вышло найти пункт и культуру, просто возвращаем ближайшую свежую перевозку
        # в этой окрестности
        sub.sort_values(["delta", "date"], ascending=[False, True])
        p = sub.tail(1)['targ_price'].to_numpy()[0]
        return p * dist, 'd_d_d_d'
    # по умолчанию, берём тариф
    return predict_4dist_by_tarif(dist, date, region_code, tarif), 't'


def prec_moving_pred(df, pr_df, tr_days, fun):
    preds = df.copy()
    preds['target_pred3'] = None
    preds['debug3'] = None
    preds['target_pred4'] = None
    preds['debug4'] = None
    preds['target_pred5'] = None
    preds['debug5'] = None
    delta = pd.Timedelta(days=tr_days + 3 - 1)
    test_delta = pd.Timedelta(days=3)
    start_date = df.index.get_level_values(0).min()
    end_date = df.index.get_level_values(0).max()
    date_range = pd.date_range(start_date + delta, end_date)
    for date in date_range:
        loc_start = start_date
        loc_end = date - test_delta
        loc_range = pd.date_range(loc_start, loc_end)
        if (loc_end + pd.Timedelta(days=3)) in df.index.get_level_values(0):
            for id, item in preds.loc[loc_end + test_delta].groupby(level=0):
                dist = item['distance'].to_numpy()[0]
                dest_id = item['dest_id'].to_numpy()[0]
                crops_id = item['crops_id'].to_numpy()[0]
                region_code = item['region_code'].to_numpy()[0]
                price, debug = fun(preds.loc[loc_start:loc_end], dist, date, crops_id, region_code, dest_id, pr_df)
                preds.loc[(loc_end + test_delta, id), 'target_pred3'] = price
                preds.loc[(loc_end + test_delta, id), 'debug3'] = debug
        if (loc_end + pd.Timedelta(days=4)) in df.index.get_level_values(0):
            for id, item in preds.loc[loc_end + pd.Timedelta(days=4)].groupby(level=0):
                dist = item['distance'].to_numpy()[0]
                # dow = item['day_of_week'].to_numpy()[0]
                dest_id = item['dest_id'].to_numpy()[0]
                # date = item['date'].to_numpy()[0]
                region_code = item['region_code'].to_numpy()[0]
                price, debug = fun(preds.loc[loc_start:loc_end], dist, date, crops_id, region_code, dest_id, pr_df)
                preds.loc[(loc_end + pd.Timedelta(days=4), id), 'target_pred4'] = price
                preds.loc[(loc_end + pd.Timedelta(days=4), id), 'debug4'] = debug
        if (loc_end + pd.Timedelta(days=5)) in df.index.get_level_values(0):
            for id, item in preds.loc[loc_end + pd.Timedelta(days=5)].groupby(level=0):
                dist = item['distance'].to_numpy()[0]
                # dow = item['day_of_week'].to_numpy()[0]
                dest_id = item['dest_id'].to_numpy()[0]
                # date = item['date'].to_numpy()[0]
                region_code = item['region_code'].to_numpy()[0]
                price, debug = fun(preds.loc[loc_start:loc_end], dist, date, crops_id, region_code, dest_id, pr_df)
                preds.loc[(loc_end + pd.Timedelta(days=5), id), 'target_pred5'] = price
                preds.loc[(loc_end + pd.Timedelta(days=5), id), 'debug5'] = debug
    return preds


class PrecedentPredictor(BaseEstimator):
    # eps - радиус окрестности (в долях от целевой дистанции) для поиска аналогов в отсутсвие точного совпадения расстояния
    def __init__(self, *, tarif_df, eps=0.1, region_code=-1):
        self.__train_df = pd.DataFrame()
        self.tarif_df = tarif_df
        self.eps = eps
        self.region_code = region_code

    def __select_if(self, df):
        if (self.region_code >= 0):
            return select_by_key(df, 'region_code', self.region_code)
        else:
            return df.copy()

    def fit(self, df, y=None):
        self.__train_df = self.__select_if(df)
        return self

    # на вход подаётся Pandas.DataFrame() со столбцами date,dist,dest_id,crops_id,region_code
    def predict(self, df):
        preds = self.__select_if(df)
        preds['pred'] = None
        # print(preds)
        for id, item in preds.iterrows():
            # print(id)
            date = item['date']
            dist = item['distance']
            dest_id = item['dest_id']
            crops_id = item['crops_id']
            region_code = item['region_code']
            price, debug = predict_by_precedent_crops(self.__train_df, dist, date, crops_id, region_code, dest_id,
                                                      self.tarif_df, self.eps)
            preds.loc[id, 'pred'] = price
            preds.loc[id, 'debug'] = debug
        return preds['pred']


class SeriesPredictor(BaseEstimator):
    # fpars - параметры фильтрации
    # mpars - параметры модели
    # region_code - код региона - обязателен, т.к. по нему определяется тариф, для которого считаются поправки
    # form_type - тип формулы 'k1' для формулы p(d)=d*k1*t(d) или 'k0k1' для формулы p(d)=k0+d*k1*t(d)
    def __init__(self, *, tarif_df, fpars, mpars, region_code, form_type, min_samples=2, min_dists=2):
        if not validate_all_pars(fpars, mpars):
            raise Exception('Incorrect parameters combination')
            return
        self.__train_df = pd.DataFrame()
        self.__fdf = pd.DataFrame()
        self.tarif_df = tarif_df
        self.fpars = fpars
        self.mpars = mpars
        self.region_code = region_code
        self.min_samples = min_samples
        self.min_dists = min_dists
        self.form_type = form_type
        if (form_type) == 'k1':
            self.__reg_fun = region_regression_series
            self.__restore_fun = predict_by_tarif_and_k
        elif (form_type) == 'k0k1':
            self.__reg_fun = region_regression_series_k0_k1
            self.__restore_fun = predict_by_tarif_and_k0_k1

    def __make_data(self, idx, dists):
        date_range = pd.date_range(idx.get_level_values(0).min(), idx.get_level_values(0).max())
        midx = pd.MultiIndex.from_product([date_range, dists], names=['date', 'distance'])
        res = pd.DataFrame(index=midx, columns=['target_pred'])
        res['distance'] = res.index.get_level_values(1)
        return res

    # число дней в наборе данных должно быть не меньше, чем
    # train_window+window-1
    # train_window - заданная параметром продолжительность обучающей выборки, в днях
    # window - заданная параметром величина окна для усреднения заявок, в днях

    def fit(self, df, y=None, *, date=None):
        self.__train_df = select_by_key(df, 'region_code', self.region_code)
        use_filter, window, filter_window, filter_order = self.fpars
        kdf = self.__reg_fun(self.__train_df, self.tarif_df, window=window, min_samples=self.min_samples, \
                             min_dists=self.min_dists, include_near=False, include_long=True,
                             region_code=self.region_code)
        if not (date is None):
            min_date = kdf.index.min()
            kdf = kdf.reindex(pd.date_range(min_date, date))
        self.__fdf = filter(kdf, use_filter, filter_window, filter_order)
        return self

    # pred_start, pred_end - начало и конец интервала предсказания в днях
    # dists - одномерный массив numpy интересующих расстояний
    def predict(self, pred_start, pred_end, *, df=None, dists=None, ):
        var, train_window, n_harm, period, poly_order = self.mpars
        pred_df = predict_series_once(self.__fdf, tr_days=train_window, pred_start=pred_start, pred_end=pred_end,
                                      var=var, \
                                      n_harm=n_harm, period=period, deg=poly_order)
        # если есть набор данных на входе - берём его, обрезаем по датам предсказаний и региону
        if (not df is None):
            min_date = pred_df.index.min()
            max_date = pred_df.index.max()
            preds = select_by_key(df.loc[min_date:max_date], 'region_code', self.region_code)
        else:
            # по умолчанию, предсказываем для концов интервалов расстояний из тарифа
            if (dists is None):
                dists = intervals_for_region(self.tarif_df, self.region_code)[:, 1]
            preds = self.__make_data(pred_df.index, dists)
        preds = self.__restore_fun(preds, self.tarif_df, self.region_code, pred_df, pred_df.columns, 'target_pred')
        return preds
