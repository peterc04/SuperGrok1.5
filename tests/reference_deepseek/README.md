# Official DeepSeek-V4.1-Flash inference reference (test oracle)

`model.py` and `engram.py` are copied **unmodified** from
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference
(MIT License, Copyright (c) DeepSeek). They are used only by
`tests/test_deepseek_reference.py`, which checks that `deepseek_v41` computes the
same function as the official code.

The other modules here are stand-ins for imports the tests do not exercise:
`kernel.py` replaces the tilelang GPU kernels with exact pure-PyTorch versions
(quantization becomes the identity), and `vision.py` / `image_processor.py` /
`sympy.py` are minimal stubs (vision is disabled; `sympy.isprime` is the only
function engram.py needs).
