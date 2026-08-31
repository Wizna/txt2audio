import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import os
import logging
import functools
from config import config
from ffmpeg_filters import build_subtitles_filter
from pathing import tmp_output_path

logger = logging.getLogger('txt2audio')
_warned_default_cover_fonts = False
_warned_missing_subtitles_filter = False


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


def _wrap_text_to_width(draw, text, font, max_width):
    if not text:
        return []
    if draw.textbbox(xy=(0, 0), text=text, font=font)[2] <= max_width:
        return [text]

    lines = []
    current = ''
    for char in text:
        candidate = current + char
        left, top, right, bottom = draw.textbbox(xy=(0, 0), text=candidate, font=font)
        if current and right - left > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


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
        global _warned_default_cover_fonts
        if not _warned_default_cover_fonts:
            logger.warning(f"无法加载视频封面自定义字体: {resources_dir}；已改用默认字体。")
            logger.debug(f"Font loading error: {e}")
            _warned_default_cover_fonts = True
        font = ImageFont.load_default()
        smaller_font = ImageFont.load_default()
        number_font = ImageFont.load_default()

    d = ImageDraw.Draw(img)

    current_h = int(max_h * 0.16)
    pad = int(max_h * 0.03)
    max_text_width = int(max_w * 0.88)
    text_bottom_limit = int(max_h * 0.72)
    for idx, sub_para in enumerate(toc.split('/')):
        sub_para = re.sub(r'（.+?）', ' ', sub_para)
        for line in sub_para.split(' '):
            line = line.strip()

            if not line:
                continue

            selected_font = font if idx == 0 else smaller_font
            for wrapped_line in _wrap_text_to_width(d, line, selected_font, max_text_width):
                (left, top, right, bottom) = d.textbbox(xy=(0, 0), text=wrapped_line, font=selected_font)
                w = right - left
                h = bottom - top
                if current_h + h > text_bottom_limit:
                    break
                d.text(
                    ((max_w - w) / 2, current_h),
                    wrapped_line,
                    font=selected_font,
                    fill=(255, 255, 255),
                )
                current_h += h + pad

    (left, top, right, bottom) = d.textbbox(xy=(0, 0), text=f'{number}', font=number_font)
    w = right - left
    h = bottom - top
    draw_underlined_text(
        d,
        ((max_w - w) / 2, int(max_h * 0.77)),
        f'{number}',
        font=number_font,
        fill=(255, 255, 255),
    )

    result = Path(audio).with_name('cover.jpg')
    img.save(result, quality=config['video'].get('cover_jpeg_quality', 70))
    return str(result)


@functools.lru_cache(maxsize=1)
def _ffmpeg_has_subtitles_filter():
    """缓存 ffmpeg 的 subtitles 滤镜检测结果，每个进程只运行一次。"""
    try:
        ret = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True, timeout=10)
        return 'subtitles' in ret.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def transform_wav_to_video(number, audio, toc, resources_dir, keep_subtitles=False, warnings=None):
    """wav + 封面图 -> mp4，成功后删除原 wav。使用临时文件确保原子写入。
    当 config.video.subtitles 为 true 且存在对应 SRT 文件时，烧入字幕。"""
    image = create_image_from_text(number=number, toc=toc, audio=audio, resources_dir=resources_dir)
    audio_path = Path(audio)
    video_path = audio_path.with_suffix('.mp4')
    tmp_video_path = tmp_output_path(video_path)
    vc = config['video']

    srt_path = audio_path.with_suffix('.srt')
    use_subtitles = vc.get('subtitles', False) and srt_path.is_file()
    if use_subtitles and not _ffmpeg_has_subtitles_filter():
        warning = {
            "warning_code": "subtitle_burn_in_skipped",
            "message": "当前 ffmpeg 缺少 subtitles 滤镜，本次运行将跳过字幕烧录。",
            "audio_file": str(audio_path),
            "subtitle_file": str(srt_path),
        }
        if warnings is not None:
            warnings.append(warning)
        global _warned_missing_subtitles_filter
        if not _warned_missing_subtitles_filter:
            logger.warning(warning["message"])
            _warned_missing_subtitles_filter = True
        use_subtitles = False

    # 字幕需要更高帧率以精确显示/消失
    framerate = max(vc.get('ffmpeg_video_framerate', 1), 10) if use_subtitles else vc.get('ffmpeg_video_framerate', 1)

    command = [
        'ffmpeg', '-y', '-loop', '1', '-i', image, '-i', str(audio_path),
        '-r', str(framerate),
        '-c:v', vc['ffmpeg_video_codec'], '-tune', vc['ffmpeg_tune'],
        '-crf', str(vc.get('ffmpeg_video_crf', 28)), '-preset', vc.get('ffmpeg_video_preset', 'medium'),
        '-c:a', vc['ffmpeg_audio_codec'], '-b:a', vc['ffmpeg_audio_bitrate'],
        '-pix_fmt', vc['ffmpeg_pixel_format'],
    ]
    if use_subtitles:
        subtitle_style = vc.get('subtitle_style', 'FontSize=20,PrimaryColour=&Hffffff,Alignment=2,MarginV=90')
        command += ['-vf', build_subtitles_filter(srt_path, subtitle_style)]
    command += ['-shortest', '-movflags', '+faststart', str(tmp_video_path)]

    logger.debug(f'ffmpeg command: {command}')
    ret = subprocess.run(command, capture_output=True)
    logger.debug(ret.stdout.decode(errors='replace'))
    if ret.returncode == 0:
        os.replace(tmp_video_path, video_path)
        os.remove(audio_path)
        # 清理已烧入视频的字幕中间文件；显式保留时不删除。
        if use_subtitles and not keep_subtitles and srt_path.is_file():
            os.remove(srt_path)
        return str(video_path)
    else:
        if tmp_video_path.is_file():
            os.remove(tmp_video_path)
        stderr = ret.stderr.decode(errors='replace').strip()
        if stderr:
            logger.debug(stderr)
        detail = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), '')
        if detail:
            raise RuntimeError(f'Video conversion failed (code {ret.returncode}): {detail}')
        raise RuntimeError(f'Video conversion failed (code {ret.returncode})')
