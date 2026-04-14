import subprocess
import shlex
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import os
import logging
from config import config

logger = logging.getLogger('txt2audio')


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
    """生成视频封面图，toc 格式为 "书名/卷名/章名"，cover.jpg 同目录复用。
    支持 portrait/landscape 方向，文字布局按比例自适应。"""
    if max_w is None:
        max_w = config['video']['width']
    if max_h is None:
        max_h = config['video']['height']

    orientation = config['video'].get('orientation', 'portrait')
    if orientation == 'landscape':
        max_w, max_h = max(max_w, max_h), min(max_w, max_h)
    else:
        max_w, max_h = min(max_w, max_h), max(max_w, max_h)

    r, g, b = get_color_from_text(s=toc.split('/')[0])
    img = Image.new('RGB', (max_w, max_h), color=(r, g, b))

    # 字体大小按短边比例缩放（config 中的值基于 720px 短边设计）
    short_edge = min(max_w, max_h)
    font_scale = short_edge / 720.0

    try:
        font = ImageFont.truetype(str(resources_dir / 'YunFengFeiYunTi-2.ttf'),
                                  int(config['video']['font_size_title'] * font_scale))
        smaller_font = ImageFont.truetype(str(resources_dir / 'YangRenDongZhuShiTi-Extralight-2.ttf'),
                                          int(config['video']['font_size_subtitle'] * font_scale))
        number_font = ImageFont.truetype(str(resources_dir / 'DTM-Mono-1.otf'),
                                         int(config['video']['font_size_number'] * font_scale))
    except (OSError, IOError) as e:
        logger.warning(f"Could not load custom fonts from {resources_dir}")
        logger.warning(f"Error: {e}")
        logger.warning("Falling back to default font. Video covers will use system default.")
        font = ImageFont.load_default()
        smaller_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    d = ImageDraw.Draw(img)

    current_h = int(max_h * 0.16)
    pad = int(max_h * 0.03)
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
    draw_underlined_text(d, ((max_w - w) / 2, int(max_h * 0.77)), f'{number}', font=number_font)

    result = f'{os.path.dirname(audio)}/cover.jpg'
    img.save(result, quality=config['video'].get('cover_jpeg_quality', 70))
    return result


def transform_wav_to_video(number, audio, toc, resources_dir):
    """wav + 封面图 -> mp4，成功后删除原 wav。使用临时文件确保原子写入。
    当 config.video.subtitles 为 true 且存在对应 SRT 文件时，烧入字幕。"""
    image = create_image_from_text(number=number, toc=toc, audio=audio, resources_dir=resources_dir)
    video_path = str(Path(audio).with_suffix('.mp4'))
    tmp_video_path = video_path.replace('.mp4', '.tmp.mp4')
    vc = config['video']

    srt_path = str(Path(audio).with_suffix('.srt'))
    use_subtitles = vc.get('subtitles', False) and os.path.isfile(srt_path)

    # 字幕需要更高帧率以精确显示/消失
    framerate = max(vc.get('ffmpeg_video_framerate', 1), 10) if use_subtitles else vc.get('ffmpeg_video_framerate', 1)

    command_line = (
        f'ffmpeg -y -loop 1 -i {shlex.quote(image)} -i {shlex.quote(audio)}'
        f' -r {framerate}'
        f' -c:v {vc["ffmpeg_video_codec"]} -tune {vc["ffmpeg_tune"]}'
        f' -crf {vc.get("ffmpeg_video_crf", 28)} -preset {vc.get("ffmpeg_video_preset", "medium")}'
        f' -c:a {vc["ffmpeg_audio_codec"]} -b:a {vc["ffmpeg_audio_bitrate"]}'
        f' -pix_fmt {vc["ffmpeg_pixel_format"]}'
    )
    if use_subtitles:
        subtitle_style = vc.get('subtitle_style', 'FontSize=22,PrimaryColour=&Hffffff,Alignment=2,MarginV=40')
        # ffmpeg subtitles 滤镜路径中需要转义特殊字符
        escaped_srt = srt_path.replace("'", r"'\''").replace(':', r'\:')
        command_line += f" -vf \"subtitles='{escaped_srt}':force_style='{subtitle_style}'\""
    command_line += f' -shortest -movflags +faststart {shlex.quote(tmp_video_path)}'

    logger.debug(f'ffmpeg command: {command_line}')
    ret = subprocess.run(command_line, capture_output=True, shell=True)
    logger.debug(ret.stdout.decode())
    if ret.returncode == 0:
        os.rename(tmp_video_path, video_path)
        os.remove(audio)
        # 清理字幕文件
        if os.path.isfile(srt_path):
            os.remove(srt_path)
    else:
        if os.path.isfile(tmp_video_path):
            os.remove(tmp_video_path)
        logger.error(f'ffmpeg failed (code {ret.returncode}), keeping {audio}')
        logger.error(ret.stderr.decode())
