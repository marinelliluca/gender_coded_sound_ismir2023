# Music Gender Classification in Children's Advertising

License: MIT

PyTorch implementation of a multi-task learning model for music gender classification in children's advertising 🎶📺

## Reference
[Paper title, Journal/Conference, Year, Link - If available]

-- [Author names]

## Summary


Data Processing: 
- Soundtracks trimmed (last 5 seconds removed), 
- features averaged across trimmed soundtrack

Embeddings: 
- MFCCs (using librosa, 20 bands, delta and delta deltas)
- MSD model (256 dimensions, re-implementation of \cite{won2019toward})
- OpenL3 embeddings (512 dimensions, \cite{cramer2019look} conda package).

Training: 
- Model checkpoint
- early stopping (patience: 30 epochs, max: 200)
- 5x 5-fold cross-validation
- AdamW optimizer

Multi-task Learning: 
- aesthetic emotions
- music perceptual features
- 4 voice-related classification tasks
- main classification of the commercials in gender targets.

## Used repositories
- https://github.com/minzwon/sota-music-tagging-models 
- https://github.com/marinelliluca/transformer-based-music-auto-tagging
