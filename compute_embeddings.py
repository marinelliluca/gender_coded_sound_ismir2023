import librosa
import torchopenl3


def compute_openl3(fn_or_y, which = 'env'):
    '''
    Wrapper for openl3. Computes the embedding of an audio file or of a numpy array.

    [1] Gyanendra Das, Humair Raj Khan, Joseph Turian (2021). torchopenl3 (version 1.0.1). 
    DOI 10.5281/zenodo.5168808, https://github.com/torchopenl3/torchopenl3.
    '''

    sr = 48000 # 48KHz as in the paper

    if type(fn_or_y) == str:
        y, sr = librosa.core.load(fn_or_y, sr=sr, res_type='kaiser_fast')
    else:
        y = fn_or_y
    
    audio_embedding, _ = torchopenl3.get_audio_embedding(y, sr, input_repr="mel128", content_type=which, 
                                                            embedding_size=512, hop_size=0.5, center=False)

    return audio_embedding.detach().cpu().numpy().squeeze()
