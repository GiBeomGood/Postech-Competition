from requests import post
from json import dumps

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJGSld0TlRjUjhvVzN6VGVUOGZBOFBlIiwiaWF0IjoxNjk4Mjk4NzkwLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.bDPjaFn4r8CYEC7bceczARApETGQyxLvKOMZLRxChf0'



def submit(y_pred):
    success = post(
        f'https://research-api.solarkim.com/cmpt-2023/bids',
        data=dumps(y_pred),
        headers={'Authorization': f'Bearer {API_KEY}'
    }).json()
    
    return success