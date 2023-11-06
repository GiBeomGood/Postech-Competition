from requests import get
import pandas as pd
from itertools import product

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJGSld0TlRjUjhvVzN6VGVUOGZBOFBlIiwiaWF0IjoxNjk4Mjk4NzkwLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.bDPjaFn4r8CYEC7bceczARApETGQyxLvKOMZLRxChf0'

today = pd.Timestamp.now()
bid_round = 1
headers = {'Authorization': f'Bearer {API_KEY}'}
link = 'https://research-api.solarkim.com/cmpt-2023'


def get_gen_forecast(today: pd.Timestamp=today):
    date = today + pd.Timedelta(days=1)
    date = date.strftime('%Y-%m-%d')
    result = get(f'{link}/gen-forecasts/{date}/{bid_round}', headers=headers).json()

    result = pd.DataFrame(result)
    result.time = pd.to_datetime(result.time).dt.tz_convert('Asia/Seoul')
    return result


def get_weather_forecast(today: pd.Timestamp=today):
    date = today + pd.Timedelta(days=1)
    date = date.strftime('%Y-%m-%d')
    result = get(f'{link}/weathers-forecasts/{date}/{bid_round}', headers=headers).json()

    result = pd.DataFrame(result)
    result.time = pd.to_datetime(result.time).dt.tz_convert('Asia/Seoul')
    return result


def get_bid_results(today: pd.Timestamp=today):
    date = today - pd.Timedelta(days=3)
    result = []

    while date < today:
        date_string = date.strftime('%Y-%m-%d')
        temp = get(f'{link}/bid-results/{date_string}', headers=headers).json()
        for i, j in product(range(24), range(1, 6)):
            temp[i][f'model{j}'] = temp[i][f'model{j}']['error']
        result.extend(temp)
        date += pd.Timedelta(days=1)

    result = pd.DataFrame(result)
    return result