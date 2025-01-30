import streamlit as st
from audio_recorder_streamlit import audio_recorder
import plotly.express as px
import plotly.graph_objects as go
import joblib
# from io import BytesIO
# from pydub import AudioSegment
# import speech_recognition as sr

import STT
from time import time, sleep
import pandas as pd
from PIL import Image
# import streamlit_nested_layout

st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1,0.3])
empty1,con3,empty2 = st.columns([0.3,1,0.3])
empty1,con4,con5,empty2 = st.columns([0.3,0.5,0.5,0.3])
empyt1,con6,con7,empty2 = st.columns([0.3,0.5,0.5,0.3])
# empyt1,con4,empty2 = st.columns([0.3,1.0,0.3])
# empyt1,con5,con6,empty2 = st.columns([0.3,0.5,0.5,0.3])


def speech_to_text(DIR: str):
    id = STT.BitoPost(DIR)
    sleep(5)
    result = STT.BitoGet(id)
    return result

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# @st.cache
def main():
    # with empty1 : 
    #     empty()
    # with empty2:
    #     empty()
    local_css("style.css")
    with con1:
        img = Image.open('title.png')
        st.image(img)

    with con3:
        st.title('Voice Phishing Detection Algorithm 🔍')
        st.markdown('💖모바일로 접속한 경우 마이크를 **한 번 더 클릭**하여 녹음을 진행하세요')
    with con4:
        st.subheader('🔴 Click to record ')
        audio_bytes = audio_recorder(
        text="",
        pause_threshold=500.0, # 100초 늘려야할듯..?
        # recording_color="#6aa36f",
        # neutral_color="#909090",
        # icon_name="volumne",
        icon_size="4x"
    )
    with con5:
        st.subheader('💡 Progress')

        if audio_bytes:
            st.markdown('⏹ Stop Recording')
            with open("audio.wav", "wb") as f:
                f.write(audio_bytes)
            st.markdown(' ⏳ Speech To Text 진행 중...')
            try:
                st.session_state.text_data = speech_to_text("audio.wav")
            except:
                st.markdown('❗ **STT 변환 실패 다시 녹음하세요**')
                raise

            st.markdown('🔧 Call Classification Model & Encoder')
            model = joblib.load('best_model.pkl')
            encoder = joblib.load('best_tfvec.pkl')
            
            st.session_state.prob=1
            result_dict = {0:0}
            slice_num = 20 #slice 할 글자 수
            for i in range(round(len(st.session_state.text_data)/slice_num)):
                text = st.session_state.text_data[ : slice_num*(1+i)]
                array = model.predict_proba(encoder.transform([text]))
                st.session_state.prob = array[0][0]
                result_dict[slice_num*(1+i)] = 1 - st.session_state.prob 
            st.markdown('🍀 Finish')
            st.session_state.df = pd.DataFrame.from_dict([result_dict]).transpose().reset_index()
            st.session_state.df.columns = ['Text Length', 'Probabilty']
        
        else:
            st.markdown("""  
            <br>
            <br/>
            """,unsafe_allow_html=True)

        with con6:
            st.subheader('📝 결과보기 ')
            if audio_bytes:
                result_prob = round(1-st.session_state.prob,3)
                if result_prob > 0.7:
                    st.image(Image.open('red.png'), width = 250)
                elif result_prob > 0.3:
                    st.image(Image.open('yellow.png'), width = 250)
                else:
                    st.image(Image.open('green.png'), width = 250)

                # st.title(f'{result_prob*100}%')
                if result_prob > 0.7:
                    st.subheader(f"📢 보이스피싱 확률이 {result_prob*100}% 입니다.")
                elif result_prob > 0.3:
                    st.subheader(f"📢 보이스피싱 확률이 {result_prob*100}% 입니다.")
                else:        
                    st.subheader(f"📢 보이스피싱 확률이 {result_prob*100}% 입니다.")

                audio_file = open("audio.wav", 'rb')
                st.audio( audio_file.read() , format='audio/wav')
                # local_css("style.css")
                with st.expander('📂 RESULT TEXT', expanded=True):
                    st.markdown(st.session_state.text_data)


        with con7:
            st.subheader('📊 Voice Phishing Probabilty')
            if audio_bytes:
                result_prob = round(1-st.session_state.prob,3)
                # area plot 색깔 지정
                color_dict={'위험':'red',
                        '경고':'orange',
                        '안전':'green'}
                if result_prob > 0.7:
                    state = "위험"
                elif result_prob > 0.3:
                    state = "경고"
                else:
                    state = "안전"
                st.session_state.df['state'] = state
                size = 500
                st.session_state.fig = px.area(st.session_state.df, x='Text Length', y='Probabilty', markers=True, color="state", color_discrete_sequence=[color_dict[state],],width=size, height=400) 
                st.session_state.fig.update_layout(
                                            paper_bgcolor = "white",
                                            showlegend=False)
                # st.session_state.fig.update_traces(hovertemplate='%{y}')
                st.session_state.fig.update_yaxes(range=[0,1])
                st.session_state.fig.update_yaxes(title_text = "")

                st.plotly_chart(st.session_state.fig, theme = "streamlit")

            else:
                st.markdown("""  
                <br>
                <br/>
                """,unsafe_allow_html=True)



if __name__ == "__main__":
    main()