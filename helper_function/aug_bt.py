import sys
import os
import time

# googletrans
from googletrans import Translator

# # pororo app key
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# import API_KEY

# papago api 
def papago_get_translate(text, In_lang, Out_lang):

    client_id = API_KEY.GET_TRANSLATE_ID()
    client_secret = API_KEY.GET_TRANSLATE_KEY()

    data = {'text' : text,       # text
            'source' : In_lang,  # Input lang
            'target': Out_lang}  # Output lang

    url = "https://openapi.naver.com/v1/papago/n2mt"

    header = {'X-Naver-Client-Id' : client_id,
              'X-Naver-Client-Secret' : client_secret}

    response = requests.post(url, headers=header, data=data)
    rescode = response.status_code

    if rescode == 200:
        send_data = response.json()
        trans_data = (send_data['message']['result']['translatedText'])
        return trans_data

    else:
        print('Error Code : {0}'.format(rescode))


# googletrans
def get_translate(text, inlang, outlang):

    translator = Translator()
    time.sleep(0.1)
    trans = translator.translate(text, src=inlang, dest=outlang)

    # print(f'translation result : {trans.text}')

    return trans.text

# Back Translation

# ko to en
def BT_ko2en(text):

    '''
    print(f'BT ko2en :')
    if len(text) >= 10: print(f'{text[:10]}')
    else: print(text)
    print(f'===>')
    '''

    try:
        out = get_translate(text, 'ko', 'en')
    except:
        print('None')
        try:    # out None
            if len(text) >= 20: print(f'raw : {text[:20]}')
            else: print(f'raw : {text}')
        except: # text None
            return None
        return None
    
    '''
    if len(out) >= 10: print(f'{out[:10]}')
    else: print(out)
    '''

    return out

# en to ko
def BT_en2ko(text):

    try:
        out = get_translate(text, 'en', 'ko')
    except:
        print('None')
        try:
            if len(text) >= 20: print(f'raw : {text[:20]}')
            else: print(f'raw : {text}')
        except: # text : None
            return None
        return None
    
    return out

# ko to jp
def BT_ko2jp(text):

    try:
        out = get_translate(text, 'ko', 'ja')
    except:
        print('None')
        try:
            if len(text) >= 20: print(f'raw : {text[:20]}')
            else: print(f'raw : {text}')
        except: # text : None
            return None
        return None

    return out

# jp to ko
def BT_jp2ko(text):

    try:
        out = get_translate(text, 'ja', 'ko')
    except:
        print('None')
        try:
            if len(text) >= 20: print(f'raw : {text[:20]}')
            else: print(f'raw : {text}')
        except: # text : None
            return None
        return None
    
    return out


''' sample '''
# get_translate('안녕하세요!', 'ko', 'en')
# out = BT_en2ko('hello!')
# print(out)