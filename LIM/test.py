import os
import re
import cv2
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ---- 基礎路徑設定 ----
ROOT_DIR = "/home/wayne/Documents/MMAU"
DEBUG_IMAGE_ROOT = "/home/wayne/Documents/MMAU_Debug_Images"
os.makedirs(DEBUG_IMAGE_ROOT, exist_ok=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
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

def parse_benchmark_txt(txt_path):
    """
    精準解析 Benchmark 文字檔，提取時序與文字描述
    """
    parsed_data = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到基準測試檔案: {txt_path}")
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 5:
                t_ai_part = parts[4].split(',')
                t_ai_val = float(t_ai_part[0])
                
                first_text_chunk = t_ai_part[1] if len(t_ai_part) > 1 else ""
                remaining_text = " ".join(parts[5:])
                full_text_desc = f"{first_text_chunk} {remaining_text}".strip()
                
                parsed_data.append([
                    parts[0],       # video_id
                    int(parts[1]),  # label (1: 有事故, 0: 無事故)
                    int(parts[2]),  # start_frame
                    int(parts[3]),  # end_frame
                    t_ai_val,       # t_ai
                    full_text_desc  # texts 事故描述
                ])
                
    return pd.DataFrame(
        parsed_data, 
        columns=['video_id', 'label', 'start_frame', 'end_frame', 't_ai', 'description']
    )

def get_frame_path(image_dir, frame_num):
    """
    安全影格路徑搜尋器 (相容 4 位與 6 位數補零)
    """
    for fname in os.listdir(image_dir):
        try:
            if int(os.path.splitext(fname)[0]) == frame_num:
                return os.path.join(image_dir, fname)
        except ValueError:
            continue
    return None

def draw_bboxes_and_get_candidates(img_path, frame_num, df_csv):
    """
    在原始解析度高清圖上繪製 BBox，並回傳該影格合法存在的所有 Track ID 清單
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, []
        
    # 過濾出當前影格的追蹤資料
    df_frame = df_csv[df_csv['frame'] == frame_num]
    candidate_ids = []
    
    for _, row in df_frame.iterrows():
        track_id = int(row['track_id'])
        if track_id == -1:
            continue
            
        candidate_ids.append(track_id)
        
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        label_name = str(row['label'])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID: {track_id} | {label_name}"
        cv2.putText(img, text, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    unique_candidates = sorted(list(set(candidate_ids)))
    return img, unique_candidates

def process_clip(image_dir, video_id_str, start_frame, end_frame, label, t_ai, description, processor, llm):
    """
    防禦型高清單影格打標核心 (內含時序滾動退避與 BBox 存在性雙重校驗機制)
    """
    # ── 防禦機制 1：負樣本快速通道 ──
    if label == 0:
        print(f"\n🟢 [安全影片跳過]: {video_id_str} | 官方標籤為 0，自動將所有物件歸類為 0。")
        print("-" * 60)
        return

    # 尋找對應的 YOLO 追蹤 CSV 檔案
    csv_path = os.path.join(image_dir, "..", f"detections_{start_frame}_{end_frame}.csv")
    if not os.path.exists(csv_path):
        print(f"[跳過] 找不到追蹤 CSV 檔案: {csv_path}")
        return
        
    df_csv = pd.read_csv(csv_path)

    # 📌 核心大改動：建立時序與物件存在性的雙重退避串列
    offsets = [15, 10, 5, 0]
    target_frame = None
    img_raw_path = None
    annotated_img = None
    candidate_list = []

    for offset in offsets:
        potential_frame = int(t_ai + offset)
        
        # 1. 時間軸邊界檢查：確保退避影格不能超越此變長片段的 end_frame 限制
        if potential_frame > end_frame:
            continue
            
        # 2. 實體影像存在性檢查
        path = get_frame_path(image_dir, potential_frame)
        if path is None:
            continue
            
        # 3. 物件框存在性檢查
        img_ann, candidates = draw_bboxes_and_get_candidates(path, potential_frame, df_csv)
        if img_ann is not None and len(candidates) > 0:
            # 完美通過所有關卡：影像存在、在邊界內、且至少有一個合法 BBox！
            target_frame = potential_frame
            img_raw_path = path
            annotated_img = img_ann
            candidate_list = candidates
            break  # 鎖定成功，立刻中斷退避迴圈

    # ── 防禦機制 2：若降級嘗試到最後的 t_ai，依然沒有任何影格同時具備影像與 BBox，則執行安全 Return ──
    if target_frame is None or not candidate_list:
        print(f"\n🟡 [雙重退避失敗跳過]: {video_id_str} | 自 t_ai+15 到 t_ai 區間內皆查無任何合法 BBox 或可用實體影像。自動忽略。")
        print("-" * 60)
        return

    # 建立目前片段專屬的視覺除錯資料夾
    safe_folder_name = video_id_str.replace('/', '_') + f"_{start_frame}_{end_frame}"
    clip_debug_dir = os.path.join(DEBUG_IMAGE_ROOT, safe_folder_name)
    os.makedirs(clip_debug_dir, exist_ok=True)

    # 將完美找到的高清檢查圖落地儲存
    out_path = os.path.join(clip_debug_dir, f"tai_frame_{target_frame}_hd.jpg")
    cv2.imwrite(out_path, annotated_img)

    # ── 📌 限制候選 ID 清單的 CoT 強硬 Prompt ──
    prompt_text = f"""You are an expert traffic accident analyst. Look at this high-resolution driving frame taken at the exact onset of a potential anomaly.
Certain traffic participants are annotated with green bounding boxes and labeled as 'ID: <number>'.

Accident Description: "{description}"

Allowed Candidate IDs: {candidate_list}

Task:
Identify which single ID from the Allowed Candidate IDs list is the primary risky object causing or directly involved in the described accident scene.
If none of the objects in the candidate list are responsible, or if the scene within the boxes appears perfectly safe, select 0.

Constraints:
- You MUST only select an ID that exists in the Allowed Candidate IDs list: {candidate_list}, or select 0. Selecting any other ID is strictly forbidden.
- Respond STRICTLY in a valid JSON format.
- DO NOT wrap the output in ```json code blocks.

Required Output Format:
{{
  "reason": "A concise one-sentence description of the risky behavior observed from the chosen ID.",
  "risky_track_id": <the chosen integer ID from the list, or 0 if safe>
}}"""

    # 餵入單張高清大圖
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": out_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # 執行 vLLM 推理
    inputs = [prepare_inputs_for_vllm(messages, processor)]
    sampling_params = SamplingParams(
        temperature=0.0,  # 徹底消除隨機性
        max_tokens=256,
        top_k=-1,
    )
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    # ─── 成果列印 ───
    print(f"\n🎬 [正樣本影片]: {video_id_str} | 最終鎖定關鍵影格: {target_frame}")
    print(f"📖 [官方文字描述]: {description}")
    print(f"🔍 [合法候選 ID 清單]: {candidate_list}")
    print(f"📸 [高清檢查圖儲存位置]: {out_path}")
    print(f"🧠 [Qwen3-VL 封閉式選擇題答案]:\n{generated_text}")
    print("-" * 60)

def process_txt(path, processor, llm):
    print(f"正在載入基準測試導航表: {path}")
    df_benchmark = parse_benchmark_txt(path)
    print(f"成功載入，共計 {len(df_benchmark)} 個短片片段待處理。")

    for idx, row in tqdm(df_benchmark.iterrows(), total=len(df_benchmark), desc="總進度"):
        video_id_str = row['video_id']  
        start_f = row['start_frame']
        end_f = row['end_frame']
        label_val = row['label']
        t_ai = row['t_ai']
        description = row['description']
        
        accident_type, folder_name = video_id_str.split('/')
        
        chosen_dataset = None
        for dataset_name in ["CAP-DATA", "DADA-DATA"]:
            potential_dir = os.path.join(ROOT_DIR, dataset_name, accident_type, folder_name, "images")
            if os.path.exists(potential_dir):
                chosen_dataset = dataset_name
                break
                
        if chosen_dataset is None:
            continue
            
        video_root_dir = os.path.join(ROOT_DIR, chosen_dataset, accident_type, folder_name)
        image_dir = os.path.join(video_root_dir, "images")

        process_clip(image_dir, video_id_str, start_f, end_f, label_val, t_ai, description, processor, llm)

if __name__ == '__main__':
    checkpoint_path = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    print(f"正在以 FP8 模式初始化 Qwen3-VL-8B ...")
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.75,
        max_model_len=5000,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )
    
    test_txt = '/home/wayne/Desktop/Thesis_code/RAM/configs/little_test.txt'
    process_txt(test_txt, processor, llm)