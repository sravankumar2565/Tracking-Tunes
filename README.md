# Tracking-Tunes-v0
A music recommender system that takes a song from user and generates similar songs based on the genre predicted.<br>
The system at its core is a neural network (currently CNN) analysing the temporal features of audio of each song in Free Music Archive (FMA) - https://github.com/mdeff/fma<br>
Many further improvements are required for the model such as using spectograms as input instead of numerical features for CNN.<br>
Current accuracy of predicting the correct genre in the top 3 prediction ~82%.
<br>
The model's architecture and training parameters are in model.py
<br>
To run and test the model for yourself, the main.py file needs to be run with Streamlit.
<br>
Note: Please add your own Spotify API key and secret to the main.py file before running to get recommendations.
