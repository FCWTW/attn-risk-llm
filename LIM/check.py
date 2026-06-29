import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import time

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    total_start = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./sample.png",
                },
                {
                    "type": "text", 
                    "text": (
                        "你是一位具備數十年經驗的「專業駕駛」。\n"
                        "請仔細審視這張行車記錄器拍下的高風險畫面，並以下列結構化的格式，生成一份「行車風險簡易診斷報告」：\n\n"
                        "1. 場景異常分析：客觀且具體地描述畫面中交通參與者的危險情境。\n"
                        "2. 事故起因診斷：剖析可能引發交通事故的潛在因果邏輯。\n"
                        "3. 安全決策建議：給出駕駛在此當下應當採取的主動防禦決策。\n\n"
                        "請直接依序輸出上述三個區塊的內容，不需任何額外的問候語或引言。"
                    )
                },
            ],
        }
    ]

    checkpoint_path = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        top_k=-1,
    )

    inference_start = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    inference_time = time.perf_counter() - inference_start
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print("\n" + "=" * 40)
        print("【Qwen3-VL 本地生成之風險報告】")
        print("=" * 40)
        print(generated_text)
    total_time = time.perf_counter() - total_start
    print(f"\nInference Time : {inference_time:.3f} seconds")
    print(f"Total Time     : {total_time:.3f} seconds")