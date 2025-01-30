import streamlit as st
from audio_recorder_streamlit import audio_recorder

# from io import BytesIO
# from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep

text_data = []

def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

def main():
    st.title('👮보이스피싱 잡아라👮')
    audio_bytes = audio_recorder("Click to record", pause_threshold=10.0)
    if audio_bytes:
        with open("audio.wav", "wb") as f:
            f.write(audio_bytes)

        text_result = speech_to_text("audio.wav")
        text_data.append(text_result)
        st.markdown(f'결과: {text_data}')


if __name__ == "__main__":
    main()