import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./test.png",
            },
            {
                "type": "text", 
                "text": """這是一張行車紀錄器拍下的危險衝突畫面。請扮演具備數十年駕駛經驗的教練，針對此畫面生成一份簡易風險報告，內容必須包含：
                        - [事故起因] 畫面中哪些物件或車輛處於危險狀態？為什麼這個場景很危險？
                        - [安全建議] 駕駛當下應該如何反應（如減速、煞車、避讓）？
                        請條列式直接回答，文字請精煉。"""
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])