import os
import pkuseg
import importlib.resources

CHINESE_PUNCTUATIONS = '，。”“～·；：？/《》*&…%¥#@！【】：’‘…、（）「」–⸺ '

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

preload_models()
# with importlib.resources.path('txt2audio.resources', 'novel_dict.txt') as user_dict_path:
#     print(f'use user dictionary at {user_dict_path}')
seg = pkuseg.pkuseg(model_name="/content/default_v2",
                    user_dict=f'{os.path.dirname(__file__)}/../resources/novel_dict.txt')

GEN_TEMP = 0.76
SPEAKER = "v2/zh_speaker_8"
silence = np.zeros(int(0.15 * SAMPLE_RATE))  # quarter second of silence


def generate_wav_for_long_form(raw_sentence):
    pieces = []
    sentences = get_text_of_fix_length(text=raw_sentence)
    for sentence in sentences:
        print(f'processing: {sentence}')
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.1,  # this controls how likely the generation is to end
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    return np.concatenate(pieces)


def split_text_by_delimiter(text,
                            delimiter=CHINESE_PUNCTUATIONS):
    results = []
    is_text = text[0] not in delimiter
    tmp_text = ''
    tmp_delimiter = ''
    for c in text:
        if c in delimiter:
            if is_text:
                results.append(tmp_text)
                tmp_text = ''
            tmp_delimiter += c
        else:
            if not is_text:
                results.append(tmp_delimiter)
                tmp_delimiter = ''
            tmp_text += c

        is_text = c not in delimiter

    if tmp_text:
        results.append(tmp_text)
    if tmp_delimiter:
        results.append(tmp_delimiter)
    return results


def get_text_of_fix_length(
        text,
        size=45,
        chinese_punctuations=CHINESE_PUNCTUATIONS):
    results = []
    segments = split_text_by_delimiter(text)
    current_length = 0
    tmp_piece = []
    for seg in segments:
        is_text = seg[0] not in chinese_punctuations

        if is_text:
            current_length += len(seg)
            if current_length >= size:
                if abs(current_length - size) < abs(current_length - len(seg) -
                                                    size):
                    tmp_piece.append(seg)
                    results.append(''.join(tmp_piece))
                    tmp_piece = []
                    current_length = 0
                else:
                    results.append(''.join(tmp_piece))
                    tmp_piece = [seg]
                    current_length = len(seg)
            else:
                tmp_piece.append(seg)
        else:
            if results and not tmp_piece:
                results[-1] += seg
            else:
                tmp_piece.append(seg)
    results.append(''.join(tmp_piece))
    return results
