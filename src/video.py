import subprocess
import shlex
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import os


def get_color_from_text(s, lightness=127):
    value = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    r = value % lightness
    value //= lightness
    g = value % lightness
    value //= lightness
    b = value % lightness
    return r, g, b


def create_image_from_text(para, max_w=720, max_h=1280):
    r, g, b = get_color_from_text(s=para.split('/')[0])
    img = Image.new('RGB', (max_w, max_h), color=(r, g, b))

    font = ImageFont.truetype(
        '/Users/huangruiming/workspace/txt2audio/resources/YunFengFeiYunTi-2.ttf',
        80)
    smaller_font = ImageFont.truetype(
        '/Users/huangruiming/workspace/txt2audio/resources/YangRenDongZhuShiTi-Extralight-2.ttf',
        70)
    d = ImageDraw.Draw(img)

    current_h, pad = 200, 40
    for idx, sub_para in enumerate(para.split('/')):
        sub_para = re.sub(r'（.+）', ' ', sub_para)
        for line in sub_para.split(' '):
            line = line.strip()

            if not line:
                continue

            selected_font = font if idx == 0 else smaller_font
            w, h = d.textsize(line, font=selected_font)
            d.text(((max_w - w) / 2, current_h), line, font=selected_font)
            current_h += h + pad

    img.save(f'{os.path.dirname(para)}/cover.png')


def transform_wav_to_video(audio, image, output_video):
    command_line = 'ffmpeg -y -r 1 -loop 1 -i pil_text_font.png -i 楔子.wav -acodec copy -r 1 -shortest -vf scale=720:1280 ep1.flv'
    subprocess.run(shlex.split(command_line))
