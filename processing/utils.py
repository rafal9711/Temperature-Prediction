from typing import Tuple

import pandas as pd
from sklearn import ensemble
import pickle
from pathlib import Path

def perform_processing(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> Tuple[float, float]:

    temperature = temperature[temperature['serialNumber'] == serial_number_for_prediction]

    df_combined = pd.concat([
        temperature.rename(columns={'value': 'temp'}),
        target_temperature.rename(columns={'value': 'target_temp'}),
        valve_level.rename(columns={'value': 'valve'})
    ])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15),label='right').mean().fillna(method='ffill')
    df_combined.dropna(inplace=True)

    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value = 20.34)
    df_combined['valve_gt'] = df_combined['valve'].shift(-1, fill_value = 75)

    df_test = df_combined.tail(1)


    #TEMPERATURE PREDICTION
    with Path('model/temperature_model.p').open('rb') as temp_file:
        temperature_model = pickle.load(temp_file)

    X_test = df_test[['temp','valve']].to_numpy()
    temp_predicted = temperature_model.predict(X_test)


    #VALVE PREDICTION
    with Path('model/valve_model.p').open('rb') as valve_file:
        valve_model = pickle.load(valve_file)

    X_test = df_test[['temp', 'valve']].to_numpy()
    valve_predicted = valve_model.predict(X_test)



    return temp_predicted, valve_predicted
