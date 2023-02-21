import subprocess
import shlex
from PIL import Image, ImageDraw, ImageFont



def create_image_from_text(text):
    img = Image.new('RGB', (100, 30), color=(73, 109, 137))

    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Hello world", font=fnt, fill=(255, 255, 0))

    img.save('pil_text_font.png')


def transform_wav_to_video(audio, image, output_video):
    command_line = 'ffmpeg -y -r 1 -loop 1 -i pil_text_font.png -i 楔子.wav -acodec copy -r 1 -shortest -vf scale=720:1280 ep1.flv'
    subprocess.run(shlex.split(command_line))
