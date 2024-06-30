import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def compare_features(features1, features2):
    similarity = 1 - cosine(features1, features2)
    return similarity * 100

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_audio_librosa.py <file1> <file2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    features1 = extract_features(file1_path)
    features2 = extract_features(file2_path)

    similarity = compare_features(features1, features2)
    print(f"Similarity: {similarity:.2f}%")
