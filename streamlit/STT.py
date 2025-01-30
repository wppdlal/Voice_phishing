import requests
import json

def get_token():
    client_id = 1 # 본인 거 발급 받아서 써야됨
    client_secret = 1 #본인 거 발급 받아서 써야됨
    resp = requests.post(
        'https://openapi.vito.ai/v1/authenticate',
        data={'client_id': client_id,
            'client_secret': client_secret}
    )
    token = resp.json()['access_token']
    return token

# file directory 설정 필수!
def BitoPost(DIR: str):
    config = {
    "diarization": {
        "use_verification": False
    },
    "use_multi_channel": False
    }

    token = get_token()

    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': 'bearer '+token},
        data={'config': json.dumps(config)},
        files={'file': open(DIR, 'rb')}
    )
    resp.raise_for_status()
    
    id = resp.json()['id']

    # resp2 = requests.get(
    #     'https://openapi.vito.ai/v1/transcribe/'+id,
    #     headers={'Authorization': 'bearer '+token},
    # )
    # resp2.raise_for_status()
    # text = resp2.json()
    return id

# id 값에 BitoPost 함수의 output 넣어야함
# STT 성능이 조금(약간의 대기시간 넣어줘야 함)
def BitoGet(id: str):
    token = get_token()
    resp2 = requests.get(
        'https://openapi.vito.ai/v1/transcribe/'+id,
        headers={'Authorization': 'bearer '+token},
    )
    resp2.raise_for_status()
    text = resp2.json()

    return ' '.join([i['msg'] for i in text['results']['utterances']])
