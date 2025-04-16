# TraceableTTS

Traceable TTS: Toward Watermark-Free TTS with Strong Traceability

## Abstract

Recent advances in Text-To-Speech (TTS) technology have enabled synthetic speech to mimic human voices with remarkable realism, raising significant security concerns. This underscores the need for traceable TTS models—systems capable of tracing their synthesized speech without compromising quality or security. However, existing methods predominantly rely on explicit watermarking on speech or on vocoder, which degrades speech quality and is vulnerable to spoofing. To address these limitations, we propose a novel framework for model attribution. Instead of embedding watermarks, we train the TTS model and discriminator using a joint training method that significantly improves traceability generalization while preserving—and even slightly improving—audio quality. This is the first work toward watermark-free TTS with strong traceability. To promote progress in related fields, we will release the code upon acceptance of the paper.

## Paper

- **arXiv**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (TODO)
  
## Checkpoints
- **Hugging Face**: [TraceableTTS on Hugging Face](https://huggingface.co/yourusername/TraceableTTS) (TODO)

## Install

```bash
pip install -r requirements.txt
```

## Training

```bash
bash gan_train.sh
```

## Acknowledgement

I would like to extend a special thanks to authors of F5-TTS, since our code base is mainly borrowed from [F5-TTS](https://github.com/SWivid/F5-TTS).
