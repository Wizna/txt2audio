# txt2audio

> Transform txt book to audio book 将网络小说转换成有声书（主要是为了保护我的眼睛）

USAGE:

- 运行示例 `cd src && python3 txt2audio.py ../demo/《英雄志》（校对第1-22卷）作者：孙晓.txt `

NOTE:

- Decoder stopped with `max_decoder_steps` 500

  - 去 ~/Library/Application Support/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST/config.json 修改 max_decoder_steps 为
  10000

  - 还一个比较常见的原因，是"，"这种结尾，导致没有正确停止

- 运行有问题先看 python 库的配置 `pip3 install -r requirements.txt`

TODO:
- [ ] 模型迁移为 faireseq - transformer

