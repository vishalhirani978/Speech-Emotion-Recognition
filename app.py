import joblib
import librosa
import soundfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")


@st.cache_resource
def load_model():
    return joblib.load("emotion_model.pkl")


model = load_model()


def extract_feature(file_obj):
    with soundfile.SoundFile(file_obj) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

    if X.ndim > 1:
        X = X.mean(axis=1)

    if sample_rate != 22050:
        X = librosa.resample(X, orig_sr=sample_rate, target_sr=22050)
        sample_rate = 22050

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    return np.hstack((mfccs, chroma, mel))


def predict_and_show(audio_file):
    audio_file.seek(0)
    st.audio(audio_file)
    audio_file.seek(0)
    features = extract_feature(audio_file).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    best = model.classes_[np.argmax(probs)]
    st.success(f"Predicted emotion: **{best}** ({probs.max() * 100:.1f}%)")
    chart = pd.DataFrame({"emotion": model.classes_, "probability": probs}).set_index("emotion")
    st.bar_chart(chart)


st.title("🎙️ Speech Emotion Recognition")
st.write("UPload audio from mic  and model will predict  emotions (calm, happy, fearful, disgust).")

tab1, tab2 = st.tabs(["Upload WAV", "Record"])

with tab1:
    uploaded = st.file_uploader("Sletect WAV file ", type=["wav"])
    if uploaded is not None:
        predict_and_show(uploaded)

with tab2:
    recorded = st.audio_input("USE MIC")
    if recorded is not None:
        predict_and_show(recorded)