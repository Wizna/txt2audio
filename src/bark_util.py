import os
import pkuseg
import importlib.resources

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
seg = pkuseg.pkuseg(model_name="default_v2", user_dict=f'{os.path.dirname(__file__)}/../resources/novel_dict.txt')

GEN_TEMP = 0.6
SPEAKER = "v2/zh_speaker_8"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence


def generate_wav_for_long_form(raw_sentence):
    pieces = []
    sentences = get_text_of_fix_length(text=raw_sentence)
    for sentence in sentences:
        print(f'processing: {sentence}')
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    return np.concatenate(pieces)


def get_text_of_fix_length(text, size=50):
    sentences = seg.cut(text)
    cnt = 0
    results = []
    tmp = ''
    for s in sentences:
        if s not in '，。“！”？：’‘… ':
            cnt += len(s)
        if cnt >= size:
            results.append(tmp)
            tmp = s
            cnt = len(s)
        else:
            tmp += s
    if tmp:
        results.append(tmp)
    return results
