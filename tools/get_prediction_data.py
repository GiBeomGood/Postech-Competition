from requests import get
import pandas as pd
from itertools import product

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJGSld0TlRjUjhvVzN6VGVUOGZBOFBlIiwiaWF0IjoxNjk4Mjk4NzkwLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.bDPjaFn4r8CYEC7bceczARApETGQyxLvKOMZLRxChf0'

today = pd.Timestamp.now()
bid_round = 1
headers = {'Authorization': f'Bearer {API_KEY}'}
link = 'https://research-api.solarkim.com/cmpt-2023'


def get_gen_forecast(start_date: pd.Timestamp=None, end_date: pd.Timestamp=None) -> pd.DataFrame:
    date = start_date
    result = []
    while date <= end_date:
        date_string = date.strftime('%Y-%m-%d')
        temp = get(f'{link}/gen-forecasts/{date_string}/{bid_round}', headers=headers).json()
        result.extend(temp)
        date += pd.Timedelta(days=1)

    result = pd.DataFrame(result)
    result.time = pd.to_datetime(result.time).dt.tz_convert('Asia/Seoul')
    return result


def get_weather_forecast(start_date: pd.Timestamp=None, end_date: pd.Timestamp=None) -> pd.DataFrame:
    date = start_date
    result = []
    while date <= end_date:
        date_string = date.strftime('%Y-%m-%d')
        temp = get(f'{link}/weathers-forecasts/{date_string}/{bid_round}', headers=headers).json()
        result.extend(temp)
        date += pd.Timedelta(days=1)

    result = pd.DataFrame(result)
    result.time = pd.to_datetime(result.time).dt.tz_convert('Asia/Seoul')
    return result


def get_gen(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date.strftime('%Y-%m-%d'))
    start_date = start_date.tz_localize('Asia/Seoul') + pd.Timedelta(hours=1)
    end_date += pd.Timedelta(days=1)
    end_date = pd.to_datetime(end_date.strftime('%Y-%m-%d')).tz_localize('Asia/Seoul')

    result = pd.read_csv('data/gens_post.csv')
    result.month = result.month.astype(str)
    result.day = result.day.apply(lambda x: f'{x:0>2d}')
    result.time = result.time.apply(lambda x: f'{x-1:0>2d}')

    result['time'] = pd.to_datetime('2023-'+result.month+'-'+result.day+' '+result.time)
    result = result.loc[:, ['time', 'amount']]
    result['time'] += pd.Timedelta(hours=1)
    result.time = result.time.dt.tz_localize('Asia/Seoul')

    result = result.loc[(result.time >= start_date) & (result.time <= end_date), :]
    result = result.reset_index(drop=True)

    return result


def get_bid_results(start_date: pd.Timestamp=None, end_date: pd.Timestamp=None) -> pd.DataFrame:
    date = start_date
    result = []

    while date <= end_date:
        date_string = date.strftime('%Y-%m-%d')
        temp = get(f'{link}/bid-results/{date_string}', headers=headers).json()
        for i, j in product(range(24), range(1, 6)):
            temp[i][f'model{j}'] = temp[i][f'model{j}']['error']
        result.extend(temp)
        date += pd.Timedelta(days=1)

    result = pd.DataFrame(result)
    result.time = pd.to_datetime(result.time).dt.tz_convert('Asia/Seoul')
    return result