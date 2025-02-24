import streamlit as st
import pandas as pd
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import spotipy
import creds
from spotipy.oauth2 import SpotifyClientCredentials
#Creds file contains Spotify API key ID and Secret
SPOTIFY_CLIENT_ID = creds.clientId
SPOTIFY_CLIENT_SECRET = creds.clientSecret
auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)
st.title("Tracking Tunes")
def recommend_songs(genres, artist=None, num_tracks=5):
    tracks = []
    for genre in genres:
        query = f"genre:{genre}"
        if artist:
            query += f" artist:{artist}"

        results = sp.search(q=query, type='track', limit=num_tracks)
        
        for item in results['tracks']['items']:
            track_name = item['name']
            track_artist = item['artists'][0]['name']
            track_url = item['external_urls']['spotify']
            track_img = item['album']['images'][0]['url']
            tracks.append((track_name, track_artist, track_url, track_img))
    
    return tracks

def load_audio_features(file_path):
    try:
        features = {}
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features['mfcc'] = list(np.mean(mfccs.T, axis=0))
        features['chroma'] = list(np.mean(chroma.T, axis=0))
        features['contrast_mean'] = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
        features['tempo'] = librosa.feature.tempo(y=y, sr=sr).mean()
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=y).mean()
        features['rms'] = librosa.feature.rms(y=y).mean()
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    except Exception as e:
        st.error(f"Error in loading file: {e}")
        return None
    
    return features

def preprocess_features(features):
    data = []
    data.append(features)
    data = pd.DataFrame(data)
    data['mfcc'] = data['mfcc'].apply(lambda x: np.array([float(i) for i in x]))
    data['chroma'] = data['chroma'].apply(lambda x: np.array([float(i) for i in x]))
    mfcc_df = pd.DataFrame(data['mfcc'].tolist(), index=data.index)
    chroma_df = pd.DataFrame(data['chroma'].tolist(), index=data.index)
    data = data.drop(['mfcc', 'chroma'], axis=1)
    data = pd.concat([data, mfcc_df, chroma_df], axis=1)
    data.columns = data.columns.astype(str)
    scaler = joblib.load('scaler.pkl')
    x = pd.DataFrame(scaler.transform(data), columns=data.columns)
    x = x.values.reshape(x.shape[0], x.shape[1], 1)
    return x

def predict_genre(x):
    model = load_model('save.keras')
    predicted_probabilities = model.predict(x)
    top_3_indices = np.argsort(predicted_probabilities[0])[-3:][::-1]
    genre_labels = joblib.load('boi.pkl')
    genre_labels = list(joblib.load('boi.pkl'))
    top_3_genres = [genre_labels[i] for i in top_3_indices]
    top_3_probabilities = predicted_probabilities[0][top_3_indices]
    return top_3_genres, top_3_probabilities


uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    file_path = f"temp_audio_file{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path, format="audio/mp3")
    
    features = load_audio_features(file_path)
    
    if features is not None:
        x = preprocess_features(features)
        top_3_genres, top_3_probabilities = predict_genre(x)
        
        st.write("Top 3 Predicted Genres:")
        for genre, prob in zip(top_3_genres, top_3_probabilities):
            st.write(f"{genre} with probability of {prob:.4f}")
        artist = st.text_input("Enter your favorite artist")
        if st.button("Recommend Songs"):
            recommended_tracks = recommend_songs(top_3_genres, artist)
            
            if not recommended_tracks:
                st.write("No tracks found.")
            else:
                for idx, track in enumerate(recommended_tracks):
                    track_name, track_artist, track_url, track_img = track
                    st.markdown(f"**{idx+1}. [{track_name} by {track_artist}]({track_url})**")
                    st.image(track_img, width=100)