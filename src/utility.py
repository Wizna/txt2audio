from typing import List, Dict
import io
from contextlib import redirect_stdout
import argparse

from TTS.api import TTS
from charset_normalizer import from_path
import re
import os
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from video import transform_wav_to_video
import math
import librosa

# NOTE: 30min 内大约读 4500 个字
CHINESE_WORD_LIMIT_HALF_HOUR = 4500
# model_name = 'tts_models/zh-CN/baker/tacotron2-DDC-GST'
model_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
tts = TTS(model_name=model_name, progress_bar=True, gpu=False)

book_delimiter = '卷章'


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def load_txt_file(file_path):
    results = from_path(file_path)
    with io.StringIO() as buf, redirect_stdout(buf):
        print(results.best())
        output = buf.getvalue()

        return output


def get_word_num(text):
    return len(re.findall(u'[\u4e00-\u9fff]', text))


def save_audio_file(wav, sample_rate, output_path: str, video_clip_index: int) -> None:
    audio_file_path = f'{output_path}-{video_clip_index}.wav'
    if os.path.isfile(audio_file_path):
        print(f"{audio_file_path} is already generated !")
        return
    stretched_audio = librosa.effects.time_stretch(y=np.array(wav, dtype=np.float32), rate=1.24,
                                                   n_fft=512)
    write(audio_file_path, sample_rate, stretched_audio)


def generate_audio_clip(text: str, output_path: str, sample_rate=22050):
    word_count = 0
    video_clip_index = 1
    wav = []
    sentences = mask_punctuations(text=text)
    # NOTE: model limit is 82
    for processed_sentences in split_long_sentences(sentences):
        wav.extend(tts.tts(text=processed_sentences, speaker_wav="./resources/female.wav", language="zh-cn",
                           speed=1.24, split_sentences=False))
        word_count += get_word_num(text=processed_sentences)

        if word_count > CHINESE_WORD_LIMIT_HALF_HOUR:
            save_audio_file(wav=wav, sample_rate=sample_rate, output_path=output_path,
                            video_clip_index=video_clip_index)
            video_clip_index += 1
            wav = []
            word_count = 0

    if wav:
        save_audio_file(wav=wav, sample_rate=sample_rate, output_path=output_path, video_clip_index=video_clip_index)
    else:
        video_clip_index -= 1

    return video_clip_index


def mask_punctuations(text):
    text = re.sub(r"([！？=@。])+", r"\1", text)  # replace ?! -> !
    text = re.sub(r"([！@=…？])\1+", r"\1", text)  # replace !! -> !
    text = re.sub(r'[…]+', '。', text)
    text = text.replace('·', '').replace('※', '')
    text = re.sub(r'[=]+', '。', text)
    text = text.replace('《', '').replace('》', '').replace("\n", " ").replace("。”", "。").replace("！”", "。").strip()

    text = re.sub(
        r"(?:https?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))*)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)?)|(?:s?ftp://(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Zd]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))(?:/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:;type=[AIDaid])?)?)|(?:news:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;/?:&=])+@(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3})))|(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)|\*))|(?:nntp://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:[a-zA-Z](?:[a-zA-Z\d]|[_.+-])*)(?:/(?:\d+))?)|(?:telnet://(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*)(?::(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?&=])*))?@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))/?)|(?:gopher://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*)(?:%09(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*)(?:%09(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))*))?)?)?)?)|(?:wais://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)(?:(?:/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)/(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))|\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;:@&=])*))?)|(?:mailto:(?:(?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+))|(?:file://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))|localhost)?/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*))|(?:prospero://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)/(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&=])*))*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[?:@&])*)))*)|(?:ldap://(?:(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?))?/(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*)(?:(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))(?:(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*))(?:(?:(?:%0[Aa])?(?:%20)*)\+(?:(?:%0[Aa])?(?:%20)*)(?:(?:(?:(?:(?:[a-zA-Z\d]|%(?:3\d|[46][a-fA-F\d]|[57][Aa\d]))|(?:%20))+|(?:OID|oid)\.(?:(?:\d+)(?:\.(?:\d+))*))(?:(?:%0[Aa])?(?:%20)*)=(?:(?:%0[Aa])?(?:%20)*))?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))*)))*))*(?:(?:(?:%0[Aa])?(?:%20)*)(?:[;,])(?:(?:%0[Aa])?(?:%20)*))?)(?:\?(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:,(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?)(?:\?(?:base|one|sub)(?:\?(?:((?:[a-zA-Z\d$\-_.+!*'(),;/?:@&=]|(?:%[a-fA-F\d]{2}))+)))?)?)?)|(?:(?:z39\.50[rs])://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*(?:\?(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?)?(?:;esn=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))?(?:;rs=(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+)(?:\+(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))+))*)?))|(?:cid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))|(?:mid:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[;?:@&=])*))?)|(?:vemmi://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:/(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&=])*)(?:(?:;(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*)=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[/?:@&])*))*))?)|(?:imap://(?:(?:(?:(?:(?:(?:(?:[a-zA-Z\d$-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+)(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))))?)|(?:(?:;[Aa][Uu][Tt][Hh]=(?:\*|(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-Fd]{2}))|[&=~])+)))(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~])+))?))@)?(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:.(?:\d+)){3}))(?::(?:\d+))?))/(?:(?:(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)?;[Tt][Yy][Pp][Ee]=(?:[Ll](?:[Ii][Ss][Tt]|[Ss][Uu][Bb])))|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:\?(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+))?(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?)|(?:(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)(?:(?:;[Uu][Ii][Dd][Vv][Aa][Ll][Ii][Dd][Ii][Tt][Yy]=(?:[1-9]\d*)))?(?:/;[Uu][Ii][Dd]=(?:[1-9]\d*))(?:(?:/;[Ss][Ee][Cc][Tt][Ii][Oo][Nn]=(?:(?:(?:[a-zA-Z\d$\-_.+!*'(),]|(?:%[a-fA-F\d]{2}))|[&=~:@/])+)))?)))?)|(?:nfs:(?:(?://(?:(?:(?:(?:(?:[a-zA-Z\d](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?)\.)*(?:[a-zA-Z](?:(?:[a-zA-Z\d]|-)*[a-zA-Z\d])?))|(?:(?:\d+)(?:\.(?:\d+)){3}))(?::(?:\d+))?)(?:(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))?)|(?:/(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Zd\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?))|(?:(?:(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*)(?:/(?:(?:(?:[a-zA-Z\d\$\-_.!~*'(),])|(?:%[a-fA-F\d]{2})|[:@&=+])*))*)?)))",
        '', text)

    if not text or not re.search(u'[\u4e00-\u9fff0-9a-zA-Z]+', text):
        return ''
    if re.search(u'[\u4e00-\u9fff]', text[-1]):
        text += '。'
    return text


def split_long_sentences(input_str, model_limit=30) -> List[str]:
    if not input_str:
        return []
    pieces = math.ceil(len(input_str) / model_limit)
    character_for_each_piece = len(input_str) // pieces
    candidates = re.split(r'([，。？！：“”])', input_str)
    result = []
    current_s = []
    for v in candidates:
        current_s.append(v)
        if not v or v in '，。？！：“”':
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


def generate_chapter(chapter_text: List, chapter_name, last_special_delimiter):
    if last_special_delimiter:
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
    # last one is special delimiter
    for i in range(start, len(chapter_structure) - 1):
        chapter_structure[i] = ''


def get_delimiter_pattern(delimiter):
    return f"(^|\s)(第[零一二三四五六七八九十]+{delimiter}|{delimiter}[零一二三四五六七八九十]+)($|\s)"


def construct_text_and_name(raw_data, book_name: str):
    table_of_contents = {}
    contents_of_chapter = {}
    toc_index = 0
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
                chapter_name = generate_chapter(chapter_text=contents, chapter_name=chapter_structure,
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
        chapter_name = generate_chapter(chapter_text=contents, chapter_name=chapter_structure,
                                        last_special_delimiter=last_special_delimiter)
        if chapter_name:
            table_of_contents[toc_index] = chapter_name
            contents_of_chapter[toc_index] = contents
            toc_index += 1

    toc_file_path = f'{os.path.dirname(__file__)}/../output/{book_name}/目录.txt'
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
    book_file_path = parse_arguments()
    assert len(book_file_path) == 1 and '.' in book_file_path[0], "输入一个文件路径，且必须包含文件后缀"
    book_name = os.path.basename(book_file_path[0]).split('.')[0]
    print(f'=========== start processing {book_name} =============')
    raw_data = load_txt_file(book_file_path[0])
    toc, contents = construct_text_and_name(raw_data=raw_data, book_name=book_name)

    span = ask_for_output_range(total=max(toc.keys()))
    for idx in span:
        if idx not in toc:
            break
        output_path = f'{os.path.dirname(__file__)}/../output/{toc[idx]}'

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        clip_num = generate_audio_clip(text=''.join(contents[idx]), output_path=output_path, sample_rate=22050)

        for i in range(clip_num):
            transform_wav_to_video(number=idx, audio=f'{output_path}-{i + 1}.wav', toc=toc[idx])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Read a text book and transform to an audio book.')
    parser.add_argument('input_file_path', metavar='input_file_path', type=str, nargs=1,
                        help='path to the text book (absolute or relative)')

    args = parser.parse_args()
    return args.input_file_path


def ask_for_output_range(total):
    var = input("请输入转换范围, (all 表示全部): \n")
    if len(var) == 0 or var == 'all':
        return range(total)
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
