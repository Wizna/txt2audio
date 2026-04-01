import subprocess
import shlex
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import os
from config import config


def draw_underlined_text(draw, pos, text, font, **options):
    (left, top, right, bottom) = draw.textbbox(xy=(0, 0), text=text, font=font)

    text_width = right - left
    text_height = bottom - top
    lx, ly = pos[0], pos[1] + text_height + 20
    draw.text(pos, text, font=font, **options)
    draw.line((lx, ly, lx + text_width, ly), width=4, **options)


def get_color_from_text(s, lightness=127):
    """标题哈希 -> 确定性 RGB，保证同一书名颜色一致。"""
    value = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    r = value % lightness
    value //= lightness
    g = value % lightness
    value //= lightness
    b = value % lightness
    return r, g, b


def create_image_from_text(number, toc, audio, resources_dir, max_w=None, max_h=None):
    """生成视频封面图，toc 格式为 "书名/卷名/章名"，cover.jpg 同目录复用。"""
    if max_w is None:
        max_w = config['video']['width']
    if max_h is None:
        max_h = config['video']['height']

    r, g, b = get_color_from_text(s=toc.split('/')[0])
    img = Image.new('RGB', (max_w, max_h), color=(r, g, b))

    try:
        font = ImageFont.truetype(str(resources_dir / 'YunFengFeiYunTi-2.ttf'), config['video']['font_size_title'])
        smaller_font = ImageFont.truetype(str(resources_dir / 'YangRenDongZhuShiTi-Extralight-2.ttf'), config['video']['font_size_subtitle'])
        number_font = ImageFont.truetype(str(resources_dir / 'DTM-Mono-1.otf'), config['video']['font_size_number'])
    except (OSError, IOError) as e:
        print(f"⚠️  Warning: Could not load custom fonts from {resources_dir}")
        print(f"Error: {e}")
        print("Falling back to default font. Video covers will use system default.")
        font = ImageFont.load_default()
        smaller_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    d = ImageDraw.Draw(img)

    current_h, pad = 200, 40
    for idx, sub_para in enumerate(toc.split('/')):
        sub_para = re.sub(r'（.+）', ' ', sub_para)
        for line in sub_para.split(' '):
            line = line.strip()

            if not line:
                continue

            selected_font = font if idx == 0 else smaller_font
            (left, top, right, bottom) = d.textbbox(xy=(0, 0), text=line, font=selected_font)
            w = right - left
            h = bottom - top
            d.text(((max_w - w) / 2, current_h), line, font=selected_font)
            current_h += h + pad

    (left, top, right, bottom) = d.textbbox(xy=(0, 0), text=f'{number}', font=number_font)
    w = right - left
    h = bottom - top
    draw_underlined_text(d, ((max_w - w) / 2, max_h - 300), f'{number}', font=number_font)

    result = f'{os.path.dirname(audio)}/cover.jpg'
    img.save(result)
    return result


def transform_wav_to_video(number, audio, toc, resources_dir):
    """wav + 封面图 -> mp4，成功后删除原 wav。"""
    image = create_image_from_text(number=number, toc=toc, audio=audio, resources_dir=resources_dir)
    video_path = str(Path(audio).with_suffix('.mp4'))
    vc = config['video']
    command_line = (
        f'ffmpeg -loop 1 -i {shlex.quote(image)} -i {shlex.quote(audio)}'
        f' -c:v {vc["ffmpeg_video_codec"]} -tune {vc["ffmpeg_tune"]}'
        f' -c:a {vc["ffmpeg_audio_codec"]} -b:a {vc["ffmpeg_audio_bitrate"]}'
        f' -pix_fmt {vc["ffmpeg_pixel_format"]}'
        f' -shortest {shlex.quote(video_path)}'
    )
    print(f'the conversion command:\n {command_line}')
    ret = subprocess.run(command_line, capture_output=True, shell=True)
    print(ret.stdout.decode())
    if ret.returncode == 0:
        os.remove(audio)
    else:
        print(f'ffmpeg failed (code {ret.returncode}), keeping {audio}')
        print(ret.stderr.decode())
