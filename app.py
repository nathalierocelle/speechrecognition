import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
from st_custom_components import st_audiorec
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import librosa
import librosa.display
from audiorecorder import audiorecorder
from flair.data import Sentence
from flair.models import SequenceTagger
from st_files_connection import FilesConnection
import boto3
import requests 
import speech_recognition as sr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bucket_name = 'speechrecognition-streamlit'
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

st.title('Speech Recognition')
with st.sidebar:
    st.subheader('About the app')
    st.text('''
            This cutting-edge app provides advanced 
            speech-to-text conversion, seamlessly  
            transforming real-time or uploaded 
            audio into transcriptions while employing 
            AI algorithms for enhanced comprehension 
            through named entity recognition.
            Coupled with sentiment analysis for 
            assessing emotional undertones, this tool 
            goes beyond audio processing, offering a 
            comprehensive solution for transforming 
            audio data into meaningful insights.
            ''')
    st.subheader('How will you import your audio data')
    import_method = st.selectbox('How will you import your audio data',
                 ['Record','Upload'], label_visibility='collapsed')
    
def save_audio(file):
    if file.size > 40000000:
        return 1
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except Exception:
        pass
    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def upload_to_aws(local_file, bucket, s3_file, aws_access_key, aws_secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)
    try:
        s3.upload_file(local_file, bucket, s3_file)
        st.write("Upload Successful")
        return True
    except FileNotFoundError:
        st.write("The file was not found")
        return False
    except NoCredentialsError:
        st.write("Credentials not available")
        return False
    
def download_file_from_s3(bucket, s3_file, aws_access_key, aws_secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)
    try:
        s3.download_file(bucket, s3_file, s3_file)
        st.write("Download Successful")
        return True
    except Exception as e:
        st.write(f"Error occurred: {e}")
        return False

# def record_audio():
#     audio = audiorecorder("Click to record", "Stop recording")
#     if len(audio) > 0:
#         # To play audio in frontend:
#         st.audio(audio.tobytes())
def record_audio():
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        # st.audio(wav_audio_data, format='audio/wav')
    # To save audio to a file:
        wav_file = open("audio/recorded_audio.wav", "wb")
        wav_file.write(wav_audio_data) 
    return "audio/recorded_audio.wav"
    
def convert_audio_to_text(file_path):
    r = sr.Recognizer()
    file_audio = sr.AudioFile(file_path)
    with file_audio as source:
        audio_text = r.record(source)
    try:
        st.write("Converting audio transcripts into text ...")
        text = r.recognize_google(audio_text)
        return text
    except Exception as e:
        st.write(f"Error occurred: {e}")
        return None

def sentiment_analysis(text):
    sentiment_analysis = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None=True)
    sentiment = sentiment_analysis(text)
    inner_list = sentiment[0]
    return sentiment

def extract_entities(text):
    tagger = SequenceTagger.load("flair/ner-english-large")
    sentence = Sentence(text)
    tagger.predict(sentence)
    return sentence.get_spans('ner')

def plot_sentiment_analysis(text_ner):
    data = [item for sublist in text_ner for item in sublist]
    categories = [item['label'] for item in data]
    values = [item['score'] for item in data]

    max_score_label = categories[values.index(max(values))]
    if max_score_label == 'positive':
        color = 'green'
    elif max_score_label == 'neutral':
        color = 'blue'
    else:
        color = 'red'

    values_percentage = [val * 100 for val in values] 
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_percentage += values_percentage[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, values_percentage, color=color, alpha=0.25)
    ax.plot(angles, values_percentage, color=color, linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{cat}: {val:.2f}%' for cat, val in zip(categories, values_percentage)])
    return fig


if __name__ == '__main__':
    if import_method == "Record":
        st.header('Record your audio')
        path = record_audio()
    
    if import_method == "Upload":
        st.header('Upload an audio file')
        uploaded_file = st.file_uploader("",type=[".mp3",".mp4",".wav","wave"])
        if uploaded_file is not None:
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success('File Uploaded in AWS')
            upload_to_aws(uploaded_file.name, bucket_name, uploaded_file.name,AWS_ACCESS_KEY, AWS_SECRET_KEY)
        if uploaded_file is not None:
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success('File Uploaded')

    
    if st.button("Transcribe"):
        converted_text = convert_audio_to_text(uploaded_file.name)
        sentiment = sentiment_analysis(converted_text)
        sentence_ner = extract_entities(converted_text)
        fig = plot_sentiment_analysis(sentiment)
            
        with st.container():
            st.header('Output')
            st.subheader('Transcript:')
            st.text(converted_text)
        with st.container():
            st.subheader('Name Entity Recognition:')
            if (sentence_ner==[]):
                st.warning('No NER tags found in the audio', icon="⚠️")
            else:
                st.text('The following NER tags are found:')
                for entity in sentence_ner:
                    st.text(entity)
        with st.container():
            st.subheader('Sentiment Analysis:')
            col1,col2 = st.columns(2)
            with col1:
                st.pyplot(fig)
                
                
    
            