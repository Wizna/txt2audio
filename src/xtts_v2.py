from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(
  text="卓凌昭站入厅心，长眉挑起，森然道：“江大人，蒙你赐帖召唤，卓某不敢不来。你若有话交代，赶紧请说吧！”江充嘿嘿一笑，道：“卓掌门别急，咱们喝上一杯，再说不迟。”说着便命人摆下桌椅，便请卓凌昭上座。",
  file_path="output.wav",
  speaker_wav="./resources/female.wav",
  language="zh-cn")

# model_name = 'tts_models/zh-CN/baker/tacotron2-DDC-GST'
# tts2 = TTS(model_name=model_name, progress_bar=True, gpu=False)
# wav = tts2.tts(text="进了知府书房，只见陆清正低头阅读自己送来的卷宗，里头详述燕陵镖局血案的来龙去脉。", speed=1.24)
# print(wav)
