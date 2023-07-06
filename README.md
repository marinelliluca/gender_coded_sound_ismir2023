# GENDER-CODED SOUND: Analysing the Gendering of Music in Toy Commercials via Multi-Task Learning
### Luca Marinelli, György Fazekas, Charalampos Saitis
#### Published @ ISMIR 2023
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of a multi-task model for a music-focused critical analysis of gender encoding strategies in toy advertising 🎶📺

##### Abstract:
Music can convey ideological stances, and gender is just one of them. Evidence from musicology and psychology research shows that gender-loaded messages can be reliably encoded and decoded via musical sounds. However, much of this evidence comes from examining music in isolation, while studies of the gendering of music within multimodal communicative events are sparse. In this paper, we outline a method to automatically analyse how music in TV advertising aimed at children may be deliberately used to reinforce traditional gender roles. Our dataset of 606 commercials included music-focused mid-level perceptual features, multimodal aesthetic emotions, and content analytical items. Despite its limited size, and because of the extreme gender polarisation inherent in toy advertisements, we obtained noteworthy results by leveraging multi-task transfer learning on our densely annotated dataset. The models were trained to categorise commercials based on their intended target audience, specifically distinguishing between masculine, feminine, and mixed audiences. Additionally, to provide explainability for the classification in gender targets, the models were jointly trained to perform regressions on emotion ratings across six scales and on mid-level musical perceptual attributes across seven scales. Standing in the context of MIR, computational social studies and critical analysis, this study may benefit not only music scholars but also advertisers, policymakers, and broadcasters.

## Demo 
<a href="https://colab.research.google.com/github/marinelliluca/gender_coded_sound_ismir2023/blob/main/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


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
- 5x (no random seed) 5-fold cross-validation, 0.1 val, 0.1 test 
- AdamW optimizer

Multi-task Learning: 
- aesthetic emotions
- music perceptual features
- 4 voice-related classification tasks
- main classification of the commercials in gender targets.
