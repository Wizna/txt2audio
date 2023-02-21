import subprocess
import shlex
from PIL import Image, ImageDraw, ImageFont
import hashlib


def get_color_from_text(s):
    value = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)
    r = value % 255
    value //= 255
    g = value % 255
    value //= 255
    b = value % 255
    return r, g, b


def create_image_from_text(text):
    r, g, b = get_color_from_text(s=text)
    img = Image.new('RGB', (100, 30), color=(r, g, b))

    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Hello world", font=fnt, fill=(255, 255, 0))

    img.save('pil_text_font.png')


def transform_wav_to_video(audio, image, output_video):
    command_line = 'ffmpeg -y -r 1 -loop 1 -i pil_text_font.png -i 楔子.wav -acodec copy -r 1 -shortest -vf scale=720:1280 ep1.flv'
    subprocess.run(shlex.split(command_line))
