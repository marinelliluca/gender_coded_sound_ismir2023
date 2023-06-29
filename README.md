# GENDER-CODED SOUND: ANALYSING THE GENDERING OF MUSIC IN TOY COMMERCIALS VIA MULTI-TASK LEARNING
## Conference: ISMIR 2023
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of a multi-task learning model for music-focused critical analysis of gender encoding strategies in toy advertising 🎶📺

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UMGSIEglIpPHSD2iCOKd72QeLUYEAI2R?usp=sharing)

## Authors
Luca Marinelli, Charalampos Saitis, George Fazekas 

## Summary

Data Processing: 
- Soundtracks trimmed (last 5 seconds removed), 
- features averaged across the trimmed soundtracks

Embeddings: 
- MFCCs (using librosa, 20 bands, delta and delta deltas)
- [MSD model](https://github.com/marinelliluca/transformer-based-music-auto-tagging) (256 dimensions)
- OpenL3 embeddings (512 dimensions).

Training and evaluation: 
- Model checkpoint
- early stopping (patience: 30 epochs, max: 200)
- 5x 5-fold cross-validation (no random seed)
- AdamW optimizer

Multi-task Learning: 
- aesthetic emotions
- music perceptual features
- 4 voice-related classification tasks
- main classification of the commercials in gender targets.
