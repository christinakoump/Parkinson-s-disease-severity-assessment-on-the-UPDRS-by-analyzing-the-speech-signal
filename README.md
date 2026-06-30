# Parkinson-s-disease-severity-assessment-on-the-UPDRS-by-analyzing-the-speech-signal
This project explores the automatic estimation of Parkinson’s disease severity from speech recordings in the PC-GITA dataset using pretrained transformer-based speech representations and deep learning methods.

## PC-GITA Parkinson Speech Analysis --  Project Summary

This project investigates automatic estimation of PD severity from speech recordings using the PC-GITA dataset. The workflow begins by indexing all audio files and merging them with the corresponding clinical metadata, including demographic information and Unified Parkinson’s Disease Rating Scale (UPDRS) scores.

Speech recordings are preprocessed by converting them to mono, resampling them to 16 kHz, and trimming them to a maximum duration of 10 seconds. Acoustic representations are then extracted using the pretrained WavLM-Base self-supervised speech model.

To improve the discriminative ability of the embeddings, the WavLM model is fine-tuned using contrastive learning. During training, triplets consisting of an anchor sample, a positive sample from the same speaker, and a negative sample from a different speaker are used with a contrastive loss that encourages embeddings from the same speaker to be closer while separating embeddings from different speakers.

After training, the model is frozen and embeddings are extracted for all recordings. Recording-level embeddings are averaged to obtain a single speaker-level representation. Finally, Ridge regression is applied to predict UPDRS scores for Parkinson’s disease participants, and performance is evaluated using Leave-One-Out cross-validation with Root Mean Squared Error (RMSE) as the evaluation metric.

## PC-GITA dataset is not included due to licensing. Users must request access separately.
## Pipeline

1. Dataset indexing (.wav file organization)
2. Metadata merging (clinical + audio data)
3. Speech preprocessing and resampling (16kHz)
4. WavLM embedding extraction
5. Contrastive learning training
6. Speaker-level embedding aggregation
7. Ridge regression for UPDRS prediction

## Model

- Backbone: Microsoft WavLM-base
- Embedding size: 768
- Loss: Contrastive loss
- Regression: Ridge Regression + LOOCV

## Installation

```bash
git clone https://github.com/yourusername/pcgita-ml.git
cd pcgita-ml
pip install -r requirements.txt
