import subprocess
import shlex
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import os


def draw_underlined_text(draw, pos, text, font, **options):
    text_width, text_height = draw.textsize(text, font=font)
    lx, ly = pos[0], pos[1] + text_height + 8
    draw.text(pos, text, font=font, **options)
    draw.line((lx, ly, lx + text_width, ly), width=4, **options)


def get_color_from_text(s, lightness=127):
    value = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    r = value % lightness
    value //= lightness
    g = value % lightness
    value //= lightness
    b = value % lightness
    return r, g, b


def create_image_from_text(number, para, max_w=720, max_h=1280):
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

    number_font = ImageFont.truetype("/Users/huangruiming/workspace/txt2audio/resources/DTM-Mono-1.otf", 40)
    w, h = d.textsize(f'{number}', font=number_font)
    draw_underlined_text(d, ((max_w - w) / 2, max_h - 100), f'{number}', font=number_font)

    result = f'{os.path.dirname(para)}/cover.jpg'
    img.save(result)
    return result


def transform_wav_to_video(number, audio):
    image = create_image_from_text(number, audio)
    video_path = audio.replace('wav', 'mp4')
    command_line = f'ffmpeg -loop 1 -i {image} -i {audio} -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest {video_path}'
    print(f'the conversion command:\n {command_line}')
    subprocess.run(shlex.split(command_line))
