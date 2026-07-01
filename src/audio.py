import torch
import torchaudio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from src.config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME,
    return_attention_mask=True
)

model = WavLMModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def load_wav(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    max_samples = TARGET_SR * MAX_SEC
    return wav[:max_samples]


def extract_embedding(path):
    wav = load_wav(path)

    inputs = feature_extractor(
        wav,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        out = model(inputs.input_values.to(device))

    emb = out.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.cpu().numpy()
