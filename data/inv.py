import librosa
import pandas as pd

df = pd.read_csv("metadata.csv")

audio_path = df.iloc[0]["audio_path"]

y, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate:", sr)
print("Duration:", len(y)/sr)

durations = []

for path in df["audio_path"]:
    y, sr = librosa.load(path, sr=None)
    durations.append(len(y)/sr)

print("Min:", min(durations))
print("Max:", max(durations))
print("Mean:", sum(durations)/len(durations))
