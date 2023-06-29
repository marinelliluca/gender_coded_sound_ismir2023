# GENDER-CODED SOUND: Analysing the Gendering of Music in Toy Commercials via Multi-Task Learning
### Luca Marinelli, Charalampos Saitis, George Fazekas
#### Published @ ISMIR 2023
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of a multi-task model for a music-focused critical analysis of gender encoding strategies in toy advertising 🎶📺

## Demo 
<a href="https://colab.research.google.com/drive/1UMGSIEglIpPHSD2iCOKd72QeLUYEAI2R?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Summary

![Tasks diagram](https://github.com/marinelliluca/gender_coded_sound_ismir2023/blob/main/tasks_diagram.png)

Data Processing: 
- Soundtracks trimmed (last 5 seconds removed), 
- features averaged across the trimmed soundtracks

Embeddings: 
- MFCCs (using librosa, 20 bands, delta and delta deltas; 60 dimensions)
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
