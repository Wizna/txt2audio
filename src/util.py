from TTS.api import TTS
from charset_normalizer import from_path
import re
import os

model_name = 'tts_models/zh-CN/baker/tacotron2-DDC-GST'
tts = TTS(model_name=model_name, progress_bar=True, gpu=False)

book_delimiter = '卷篇章回节'


def load_txt_file(file_path):
    results = from_path(file_path)

    return str(results.best())


def generate_audio_clip(text, output_path='test.wav'):
    text = mask_punctuations(text=text)
    tts.tts_to_file(text=text, file_path=output_path)


def mask_punctuations(text):
    text = text.replace('“', '，').replace('”', '，')
    text = text.replace('…', '，')
    text = text.replace('·', '，')

    return text


def generate_chapter(chapter_text, chapter_name):
    combined_name = '/'.join([i for i in chapter_name if i.strip()])
    output_path = f'./output/{chapter_name}.wav'
    generate_audio_clip(chapter_text, output_path=output_path)


def construct_text_and_name(raw_data, book_name: str):
    chapter_structure = [book_name] + ['' for _ in book_delimiter]
    contents = []

    for line in raw_data.split('\n'):
        line = line.strip()
        if not line:
            continue
        for idx, delimiter in enumerate(book_delimiter):
            pattern = f"第[\w]+{delimiter}"
            x = re.search(pattern, line)
            if x:
                if contents:
                    generate_chapter('\n'.join(contents), chapter_name=chapter_structure)
                matched_chapter_name = x.group()
                chapter_structure[idx + 1] = matched_chapter_name
                contents = []
                break
        else:
            contents.append(line)


def process(book_file_path):
    book_name = os.path.basename(book_file_path)
    raw_data = load_txt_file(book_file_path)
    construct_text_and_name(raw_data=raw_data, book_name=book_name)
