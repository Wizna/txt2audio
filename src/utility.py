from typing import List, Dict
import sys
import argparse

from charset_normalizer import from_path
from pypinyin import pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone
import re
import os
from pathlib import Path
import math
import torch
import torchaudio
from video import transform_wav_to_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'
OUTPUT_DIR = PROJECT_ROOT / 'output'

# CosyVoice submodule needs to be on sys.path for its internal imports
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'CosyVoice'))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'))

MODEL_DIR = str(PROJECT_ROOT / 'pretrained_models' / 'Fun-CosyVoice3-0.5B')
COSYVOICE_DIR = PROJECT_ROOT / 'third_party' / 'CosyVoice'
SPEAKER_WAV = str(RESOURCES_DIR / 'my_speaker.wav')
PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'  # zero-shot TTS prompt

CHINESE_WORD_LIMIT_HALF_HOUR = 6300  # 每个音频片段的汉字上限（约30分钟）
_tts = None


def get_tts():
    global _tts
    if _tts is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        _tts = AutoModel(model_dir=MODEL_DIR)
    return _tts

book_delimiter = '卷章'  # 按字符迭代，依次匹配卷、章


def load_txt_file(file_path):
    results = from_path(file_path)  # 自动检测文件编码
    return str(results.best())


def get_word_num(text):
    return len(re.findall(u'[\u4e00-\u9fff]', text))


def save_audio_file(wav_tensor, sample_rate, output_path: str, video_clip_index: int, export_indices: List) -> None:
    if wav_tensor is None or wav_tensor.numel() == 0:
        return
    export_indices.append(video_clip_index)
    audio_file_path = f'{output_path}-{video_clip_index}.wav'
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    torchaudio.save(audio_file_path, wav_tensor, sample_rate)


def check_export_file_exists(output_path, video_clip_index):
    """返回 True 表示需要导出（文件不存在），用于断点续生成。"""
    wav_path = f'{output_path}-{video_clip_index}.wav'
    mp4_path = f'{output_path}-{video_clip_index}.mp4'
    export = not (os.path.isfile(wav_path) or os.path.isfile(mp4_path))
    if not export:
        existing = mp4_path if os.path.isfile(mp4_path) else wav_path
        print(f"{existing} is already generated !")

    return export


def generate_audio_clip(text: str, output_path: str, sample_rate=None):
    """将一章文本转为音频，按字数上限切分为多个 ~30min 片段。"""
    cosyvoice = get_tts()
    if sample_rate is None:
        sample_rate = cosyvoice.sample_rate

    word_count = 0
    video_clip_index = 1
    exported_clip_indices = []
    wav_chunks = []
    sentences = annotate_polyphones(text)
    sentences = mask_punctuations(text=sentences)
    export = check_export_file_exists(output_path=output_path, video_clip_index=video_clip_index)

    for processed_sentences in split_long_sentences(sentences):
        if export:
            for chunk in cosyvoice.inference_zero_shot(
                processed_sentences, PROMPT_TEXT, SPEAKER_WAV, stream=False
            ):
                wav_chunks.append(chunk['tts_speech'])
        word_count += get_word_num(text=processed_sentences)

        if word_count > CHINESE_WORD_LIMIT_HALF_HOUR:
            combined = torch.cat(wav_chunks, dim=-1) if wav_chunks else None
            save_audio_file(combined, sample_rate, output_path, video_clip_index, exported_clip_indices)
            video_clip_index += 1
            wav_chunks = []
            word_count = 0
            export = check_export_file_exists(output_path=output_path, video_clip_index=video_clip_index)

    combined = torch.cat(wav_chunks, dim=-1) if wav_chunks else None
    save_audio_file(combined, sample_rate, output_path, video_clip_index, exported_clip_indices)
    return exported_clip_indices


def mask_punctuations(text):
    text = re.sub(r'[\u201c\u201d\u2018\u2019]', '', text)  # remove Chinese quotes “”''
    text = re.sub(r"([！？=@。])+", r"\1", text)  # replace ?! -> !
    text = re.sub(r"([！@=…？])\1+", r"\1", text)  # replace !! -> !
    text = re.sub(r'[…]+', '。', text)
    text = text.replace('·', '').replace('※', '')
    text = re.sub(r'[=]+', '。', text)
    text = text.replace('《', '').replace('》', '').replace("\n", " ").strip()

    # 移除 URL
    text = re.sub(
        r"(?:https?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))*)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)?)|(?:s?ftp://(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Zd]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:;type=[AIDaid])?)?)|(?:news:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;/?:&=])+@(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3})))|(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)|\*))|(?:nntp://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)(?:/(?:\d+))?)|(?:telnet://(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))/?)|(?:gopher://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*)(?:%09(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:%09(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*))?)?)?)?)|(?:wais://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)(?:(?:/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))|\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)|(?:mailto:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+))|(?:file://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))|localhost)?/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*))|(?:prospero://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)))*)|(?:ldap://(?:(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))?/(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*)(?:(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*))*(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))?)(?:\?(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:,(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?)(?:\?(?:base|one|sub)(?:\?(?:((?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+)))?)?)?)|(?:(?:z39\.50[rs])://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*(?:\?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?)?(?:;esn=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?(?:;rs=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?))|(?:cid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))|(?:mid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))?)|(?:vemmi://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&=])*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*))*))?)|(?:imap://(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+)(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))))?)|(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-Fd]{2}))|[&=~])+)))(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))?))@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:.(?:\d+)){3}))(?::(?:\d+))?))/(?:(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)?;[Tt][Yy][Pp][Ee]=(?:[Ll](?:[Ii][Ss][Tt]|[Ss][Uu][Bb])))|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+))?(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?)|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?(?:/;[Uu][Ii][Dd]=(?:[1-9]\d*))(?:(?:/;[Ss][Ee][Cc][Tt][Ii][Oo][Nn]=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)))?)))?)|(?:nfs:(?:(?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))?)|(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Zd\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?))|(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))",
        '', text)

    if not text or not re.search(u'[\u4e00-\u9fff0-9a-zA-Z]+', text):
        return ''
    if re.search(u'[\u4e00-\u9fff]', text[-1]):  # 确保以句号结尾，TTS 需要
        text += '。'
    return text


# 从 CosyVoice3Tokenizer 的 additional_special_tokens 提取的合法拼音 token 集合
# (来源: third_party/CosyVoice/cosyvoice/tokenizer/tokenizer.py:295-307)
_VALID_PINYIN_TOKENS = {
    # 声母
    'b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
    'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh',
    # 韵母 (无声调)
    'a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'i',
    'ian', 'in', 'ing', 'iu', 'o', 'ong', 'ou', 'u', 'uang', 'ue',
    'un', 'uo',
    # 带声调韵母 — i 系列
    'ià', 'iàn', 'iàng', 'iào', 'iá', 'ián', 'iáng', 'iáo',
    'iè', 'ié', 'iòng', 'ióng', 'iù', 'iú',
    'iā', 'iān', 'iāng', 'iāo', 'iē', 'iě', 'iōng', 'iū',
    'iǎ', 'iǎn', 'iǎng', 'iǎo', 'iǒng', 'iǔ',
    # 带声调韵母 — u 系列
    'uà', 'uài', 'uàn', 'uàng', 'uá', 'uái', 'uán', 'uáng',
    'uè', 'ué', 'uì', 'uí', 'uò', 'uó',
    'uā', 'uāi', 'uān', 'uāng', 'uē', 'uě', 'uī', 'uō',
    'uǎ', 'uǎi', 'uǎn', 'uǎng', 'uǐ', 'uǒ',
    # 带声调韵母 — v (ü) 系列
    'vè',
    # 带声调韵母 — 独立元音
    'à', 'ài', 'àn', 'àng', 'ào', 'á', 'ái', 'án', 'áng', 'áo',
    'è', 'èi', 'èn', 'èng', 'èr', 'é', 'éi', 'én', 'éng', 'ér',
    'ì', 'ìn', 'ìng', 'í', 'ín', 'íng',
    'ò', 'òng', 'òu', 'ó', 'óng', 'óu',
    'ù', 'ùn', 'ú', 'ún',
    'ā', 'āi', 'ān', 'āng', 'āo', 'ē', 'ēi', 'ēn', 'ēng',
    'ě', 'ěi', 'ěn', 'ěng', 'ěr',
    'ī', 'īn', 'īng', 'ō', 'ōng', 'ōu', 'ū', 'ūn',
    'ǎ', 'ǎi', 'ǎn', 'ǎng', 'ǎo', 'ǐ', 'ǐn', 'ǐng',
    'ǒ', 'ǒng', 'ǒu', 'ǔ', 'ǔn',
    # 独立 ü 韵母
    'ǘ', 'ǚ', 'ǜ',
}


# 需要拼音标注的高频多音字（仅标注 TTS 经常读错的字，避免过度标注影响自然度）
_POLYPHONE_CHARS = set('校')

# pypinyin 自定义词典：纠正 pypinyin 消歧错误的词组
from pypinyin import load_phrases_dict
load_phrases_dict({
    '精校': [['jīng'], ['jiào']],
    '校对': [['jiào'], ['duì']],
    '校勘': [['jiào'], ['kān']],
})


def annotate_polyphones(text: str) -> str:
    """对多音字注入 CosyVoice3 拼音 token，如 '给予' → '[j][ǐ]予'。"""
    if not text:
        return text

    # 整句上下文消歧
    tone_readings = pinyin(text, style=Style.TONE, heteronym=False, strict=True)

    result = []
    text_pos = 0
    for reading_entry in tone_readings:
        val = reading_entry[0]
        char = text[text_pos]

        # 非汉字：pypinyin 把连续非汉字合并为一个条目
        if not ('\u4e00' <= char <= '\u9fff'):
            result.append(val)
            text_pos += len(val)
            continue

        text_pos += 1

        # 非多音字
        if char not in _POLYPHONE_CHARS:
            result.append(char)
            continue

        # 轻声不标注（无声调符号，如 'de', 'le'）
        if not any(c in val for c in 'āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ'):
            result.append(char)
            continue

        initial = to_initials(val, strict=False)
        final = to_finals_tone(val, strict=False)

        # ü → v fallback (CosyVoice3 用 v 表示复合韵母中的 ü)
        if final not in _VALID_PINYIN_TOKENS:
            final_v = final.replace('ü', 'v')
            if final_v in _VALID_PINYIN_TOKENS:
                final = final_v

        # 验证 token 合法性
        if initial and initial not in _VALID_PINYIN_TOKENS:
            result.append(char)
            continue
        if final and final not in _VALID_PINYIN_TOKENS:
            result.append(char)
            continue

        tokens = []
        if initial:
            tokens.append(f'[{initial}]')
        if final:
            tokens.append(f'[{final}]')

        result.append(''.join(tokens) if tokens else char)

    return ''.join(result)


def split_long_sentences(input_str, model_limit=200) -> List[str]:
    """在中文标点处切分文本，使每段不超过 model_limit 字符。"""
    if not input_str:
        return []
    pieces = math.ceil(len(input_str) / model_limit)
    character_for_each_piece = len(input_str) // pieces
    candidates = re.split(r'([，。？！：])', input_str)
    result = []
    current_s = []
    for v in candidates:
        current_s.append(v)
        if not v or v in '，。？！：':
            continue
        possible = ''.join(current_s)
        if len(possible) > character_for_each_piece:
            if len(current_s) > 1:
                result.append(''.join(current_s[:-1]))
                current_s = [v]
            else:
                result.append(v)
                current_s = []

    if current_s:
        result.append(''.join(current_s))
    return result


def generate_chapter(chapter_name, last_special_delimiter):
    """从 chapter_structure 生成输出路径，如 "书名/第一卷/第一章"。"""
    if last_special_delimiter:  # 序/楔子/后记等，包含末位 special slot
        combined_name = '/'.join([i for i in chapter_name if i])
    else:
        combined_name = '/'.join([i for i in chapter_name[:-1] if i])

    return combined_name


def check_special_delimiter(text):
    for sub_text in text.split(' '):
        for p in ['序', '序章', '序言', '前言', '楔子', '引言', '后记', '终章']:
            if p == sub_text:
                return p

    return ''


def empty_structure(chapter_structure, start):
    # 高层级变化时清空低层级，如新卷时清空章名（末位 special slot 保留）
    for i in range(start, len(chapter_structure) - 1):
        chapter_structure[i] = ''


def get_delimiter_pattern(delimiter):
    return f"(^|\s)(第[零一二三四五六七八九十]+{delimiter}|{delimiter}[零一二三四五六七八九十]+)($|\s)"


def construct_text_and_name(raw_data, book_name: str):
    table_of_contents = {}
    contents_of_chapter = {}
    toc_index = 0
    # [书名, 卷, 章, special_delimiter]，用于拼接输出路径
    chapter_structure = [book_name] + ['' for _ in book_delimiter] + ['']
    contents = []
    input_text_lines = re.split('\r\n|\n', raw_data)
    last_special_delimiter = False

    for line in input_text_lines:
        line = line.strip()

        if not line:
            continue

        new_chapter = False
        special_delimiter = check_special_delimiter(line)
        if special_delimiter:
            new_chapter = True
            chapter_structure[-1] = special_delimiter

        for idx, delimiter in enumerate(book_delimiter):
            pattern = get_delimiter_pattern(delimiter)
            x = re.search(pattern, line)
            if x:
                matched_chapter_name = x.group()
                if chapter_structure[idx + 1] != matched_chapter_name.strip():
                    # NOTE: 有时候文章中会插入卷/章节，如果和之前没有变化，那么就继续
                    new_chapter = True
                    break

        if new_chapter:
            if contents:
                chapter_name = generate_chapter(chapter_name=chapter_structure,
                                                last_special_delimiter=last_special_delimiter)
                if chapter_name:
                    table_of_contents[toc_index] = chapter_name
                    contents_of_chapter[toc_index] = contents
                    toc_index += 1
                last_special_delimiter = False
                contents = []

            for idx, delimiter in enumerate(book_delimiter):
                pattern = get_delimiter_pattern(delimiter)
                x = re.search(pattern, line)
                if x:
                    matched_chapter_name = x.group()
                    chapter_structure[idx + 1] = matched_chapter_name.strip()
                    # NOTE: 前提是先卷，后章
                    empty_structure(chapter_structure, start=idx + 2)
        else:
            contents.append(line)

        if special_delimiter:
            last_special_delimiter = True

    if contents:
        chapter_name = generate_chapter(chapter_name=chapter_structure,
                                        last_special_delimiter=last_special_delimiter)
        if chapter_name:
            table_of_contents[toc_index] = chapter_name
            contents_of_chapter[toc_index] = contents
            toc_index += 1

    toc_file_path = str(OUTPUT_DIR / book_name / '目录.txt')
    save_table_of_contents(file_path=toc_file_path, table_of_contents=table_of_contents)

    return table_of_contents, contents_of_chapter


def save_table_of_contents(file_path, table_of_contents: Dict):
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w+') as f:
        for k, v in table_of_contents.items():
            w = f'{k:>5}:{v} \n'
            print(w)
            f.write(w)


def cli_main_process():
    args = parse_arguments()
    book_file_path = args.input_file_path
    assert len(book_file_path) == 1 and '.' in book_file_path[0], "输入一个文件路径，且必须包含文件后缀"
    book_name = Path(book_file_path[0]).stem
    if not os.path.isfile(book_file_path[0]):
        print("输入的文件路径不是一个文件，请检查文件路径！")
        return
    print(f'=========== start processing {book_name} =============')
    raw_data = load_txt_file(book_file_path[0])
    toc, contents = construct_text_and_name(raw_data=raw_data, book_name=book_name)

    # 找到最后已生成的章节，提示用户断点位置
    for idx in sorted(toc.keys(), reverse=True):
        output_path = str(OUTPUT_DIR / toc[idx])
        if os.path.isfile(f'{output_path}-1.wav') or os.path.isfile(f'{output_path}-1.mp4'):
            print(f'Last generated chapter is {idx}: {output_path}')
            break

    if not toc:
        print("未解析到任何章节，请检查文件内容！")
        return

    if args.range is not None:
        span = parse_range_string(args.range, total=max(toc.keys()))
    else:
        span = ask_for_output_range(total=max(toc.keys()))
    for idx in span:
        if idx not in toc:
            break
        output_path = str(OUTPUT_DIR / toc[idx])

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        clip_num = generate_audio_clip(text=''.join(contents[idx]), output_path=output_path)

        if args.video:
            for i in clip_num:
                transform_wav_to_video(number=idx, audio=f'{output_path}-{i}.wav', toc=toc[idx],
                                       resources_dir=RESOURCES_DIR)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Read a text book and transform to an audio book.')
    parser.add_argument('input_file_path', metavar='input_file_path', type=str, nargs=1,
                        help='path to the text book (absolute or relative)')
    parser.add_argument('--video', action='store_true',
                        help='generate MP4 video with cover image (default: audio only)')
    parser.add_argument('--range', type=str, default=None,
                        help='chapter range, e.g. "all", "8", "0~8" (skips interactive prompt)')

    return parser.parse_args()


def parse_range_string(var, total):
    if len(var) == 0 or var == 'all':
        return range(total + 1)
    else:
        indices = re.split('[~-]', var)
        assert len(indices) in (1, 2), "请输入单个数字或者一个范围, e.g. 8 or 0~8"
        if len(indices) == 1:
            s = int(indices[0])
            return range(s, s + 1)
        else:
            s = int(indices[0])
            e = int(indices[1])
            return range(s, e + 1)


def ask_for_output_range(total):
    var = input("请输入转换范围, (all 表示全部): \n")
    return parse_range_string(var, total)
