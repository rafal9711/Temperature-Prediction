import json
import pickle
import random
from pathlib import Path
from sklearn import datasets, ensemble
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics

from sklearn.linear_model import LinearRegression


def read_temp_mid_sn() -> int:
    with open('./../data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']

    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == 'temperature_middle'][0]

    return sn_temp_mid

def project_check_data():
    sn_temp_mid = read_temp_mid_sn()

    df_temp = pd.read_csv('./../data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')

    df_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp.drop(columns=['unit'],inplace=True)
    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    df_temp.set_index('time',inplace=True)
    # print(df_temp.head(5))

    df_target_temp = pd.read_csv('./../data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)
    df_target_temp['time'] = pd.to_datetime(df_target_temp['time'])
    df_target_temp.drop(columns=['unit'],inplace=True)
    df_target_temp.set_index('time',inplace=True)

    df_valve = pd.read_csv('./../data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_valve.rename(columns={'value': 'valve'}, inplace=True)
    df_valve['time'] = pd.to_datetime(df_valve['time'])
    df_valve.drop(columns=['unit'],inplace=True)
    df_valve.set_index('time',inplace=True)

    df_combined = pd.concat([df_temp,df_target_temp,df_valve])
    df_combined = df_combined.resample(pd.Timedelta(minutes=15),label='right').mean().fillna(method='ffill')
    df_combined.dropna(inplace=True)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value = 20.34)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value = 75)

    df_train1 = pd.DataFrame()
    df_train2 = pd.DataFrame()
    df_train3 = pd.DataFrame()

    for i in range(26,29):
        start = f'2020-10-{i} 04:00:00+00:00'
        stop  = f'2020-10-{i} 16:00:00+00:00'

        df_combined_resampled = df_combined.loc[start:stop]
        df_train1 = pd.concat([df_train1,df_combined_resampled])

    for i in range(19, 24):
        start = f'2020-10-{i} 04:00:00+00:00'
        stop = f'2020-10-{i} 16:00:00+00:00'

        df_combined_resampled = df_combined.loc[start:stop]
        df_train2 = pd.concat([df_train2, df_combined_resampled])

    for i in range(13, 17):
        start = f'2020-10-{i} 04:00:00+00:00'
        stop = f'2020-10-{i} 16:00:00+00:00'

        df_combined_resampled = df_combined.loc[start:stop]
        df_train3 = pd.concat([df_train3, df_combined_resampled])


    df_train_temp = pd.concat([df_train3,df_train2,df_train1])

    X_train = df_train_temp[['temp','valve']].to_numpy()[1:-1]
    y_train = df_train_temp['temp_gt'].to_numpy()[1:-1]

    reg_rf = LinearRegression()
    reg_rf.fit(X_train, y_train)

    # reg_rf = ensemble.RandomForestRegressor(random_state=42)
    # reg_rf.fit(X_train,y_train)

    with open('temperature_model.p', 'wb') as temp_file:
        pickle.dump(reg_rf, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
        print("zapisano temp")

    mask = (df_combined.index >= '2020-10-13') & (df_combined.index <= '2020-10-28')
    df_train_valve = df_combined.loc[mask]

    X_train = df_train_valve[['temp','valve']].to_numpy()[1:-1]
    y_train = df_train_valve['valve_gt'].to_numpy()[1:-1]
    #
    reg_rf = LinearRegression()
    reg_rf.fit(X_train, y_train)

    # reg_rf = ensemble.RandomForestRegressor(random_state=42)
    # reg_rf.fit(X_train,y_train)

    with open('valve_model.p', 'wb') as valve_file:
        pickle.dump(reg_rf, valve_file, protocol=pickle.HIGHEST_PROTOCOL)
        print("zapisano valve")


def main():
    random.seed(42)

    pd.options.display.max_columns = None
    project_check_data()

if __name__== '__main__':
    main()
