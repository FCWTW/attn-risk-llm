# Deployment details for LLM Inference Module
```bash
# In conda env python 3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/huggingface/transformers accelerate
pip install "qwen-vl-utils[decord]>=0.0.14"
pip install "vllm==0.11.0"
```

Use the following command to test your environment:
```bash
python3 .check.py
```