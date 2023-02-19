# txt2audio
transform txt book to audio book


NOTE:

- Decoder stopped with `max_decoder_steps` 500
  
  去 ~/Library/Application Support/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/config.json 修改 max_decoder_steps 为 10000

- max_decoder_steps 还一个比较常见的原因，是"，"这种结尾，导致没有正确停止

