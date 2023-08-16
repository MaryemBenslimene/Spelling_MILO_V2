import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sileromodels.src.silero import (silero_stt, silero_tts)
from glob import glob
from sileromodels.src.silero.utils import (init_jit_model, 
                       split_into_batches,
                       read_audio,
                       read_batch,
                       prepare_model_input)

import pickle

device = torch.device('cpu')   # you can use any pytorch device
models = OmegaConf.load('sileromodels\models.yml')


def getASRModel(language: str) -> nn.Module:

    if language == 'de':

        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='de',
                                               device=torch.device('cpu'))

    elif language == 'en':
        
        model, decoder = init_jit_model(models.stt_models.en.latest.jit, device=device)

    elif language == 'fr':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='fr',
                                               device=torch.device('cpu'))

    return (model, decoder)


def getTTSModel(language: str) -> nn.Module:

    if language == 'de':

        speaker = 'thorsten_v2'  # 16 kHz
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)

    elif language == 'en':
        model, symbols, sample_rate, example_text, apply_tts = silero_tts(language='en', speaker = 'lj_16khz')
    else:
        raise ValueError('Language not implemented')

    return model


def getTranslationModel(language: str) -> nn.Module:
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    if language == 'de':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        # Cache models to avoid Hugging face processing
        with open('translation_model_de.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_de.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
    else:
        raise ValueError('Language not implemented')

    return model, tokenizer
