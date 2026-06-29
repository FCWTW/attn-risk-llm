import os
import re
import cv2
import json
import torch
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ---- 基礎路徑設定 ----
ROOT_DIR = "/home/wayne/Documents/MMAU"
DEBUG_IMAGE_ROOT = "/media/wayne/27CD255760735841/MMAU_Debug"
os.makedirs(DEBUG_IMAGE_ROOT, exist_ok=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- 設置日誌處理 ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_labeling.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def process_clip(image_dir, npz_path, video_id_str, start_frame, end_frame, label, t_ai, description, processor, llm):
    """
    防禦型高清單影格打標核心 (內含時序滾動退避與 BBox 存在性雙重校驗機制)
    """
    # ── 防禦機制 1：負樣本快速通道 ──
    if label == 0:
        logger.info(f"[安全影片跳過]: {video_id_str} | 官方標籤為 0，自動將所有物件歸類為 0。")
        return

    csv_path = os.path.join(image_dir, "..", f"detections_{start_frame}_{end_frame}.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"[跳過] 找不到追蹤 CSV 檔案: {csv_path}")
        return
        
    df_csv = pd.read_csv(csv_path)

    offsets = [15, 10, 5, 0]
    target_frame = None
    img_raw_path = None
    annotated_img = None
    candidate_list = []

    for offset in offsets:
        potential_frame = int(t_ai + offset)
        
        if potential_frame > end_frame:
            continue
            
        path = get_frame_path(image_dir, potential_frame)
        if path is None:
            continue
            
        img_ann, candidates = draw_bboxes_and_get_candidates(path, potential_frame, df_csv)
        if img_ann is not None and len(candidates) > 0:
            target_frame = potential_frame
            img_raw_path = path
            annotated_img = img_ann
            candidate_list = candidates
            break

    # ── 防禦機制 2 ──
    if target_frame is None or not candidate_list:
        logger.warning(f"🟡 [雙重失敗跳過]: {video_id_str} | 自 t_ai+15 到 t_ai 區間內皆查無任何合法 BBox 或可用實體影像。")
        return

    safe_folder_name = video_id_str.replace('/', '_') + f"_{start_frame}_{end_frame}"
    clip_debug_dir = os.path.join(DEBUG_IMAGE_ROOT, safe_folder_name)
    os.makedirs(clip_debug_dir, exist_ok=True)

    out_path = os.path.join(clip_debug_dir, f"tai_frame_{target_frame}_hd.jpg")
    cv2.imwrite(out_path, annotated_img)

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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": out_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    inputs = [prepare_inputs_for_vllm(messages, processor)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        top_k=-1,
    )
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    # 解析大模型輸出的 JSON 結果
    risky_track_id = 0
    reason = "None"
    try:
        cleaned_text = generated_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        res_json = json.loads(cleaned_text.strip())
        risky_track_id = int(res_json.get("risky_track_id", 0))
        reason = res_json.get("reason", "")
    except Exception as e:
        logger.error(f"解析模型 JSON 輸出失敗: {e}. 原始文本: {generated_text}")

    logger.info(f"[正樣本影片]: {video_id_str}_{start_frame}_{end_frame} | 最終鎖定: {target_frame} | [候選 ID 清單]: {candidate_list} | [結果]: ID={risky_track_id}, 原因={reason}")
    # logger.info(f"📖 [官方文字描述]: {description}")
    # logger.info(f"🔍 [合法候選 ID 清單]: {candidate_list}")
    # logger.info(f"📸 [高清檢查圖儲存位置]: {out_path}")
    # logger.info(f"🧠 [Qwen3-VL 推理結果]: ID={risky_track_id} | 原因={reason}")

    # ── 📌 任務 1：完成 #TODO 修正 .npz 中的風險標籤 ──
    if risky_track_id > 0 and os.path.exists(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz_data:
                feature = npz_data['feature']
                detection = npz_data['detection']
                vid_id = npz_data['vid_id']
                toa = npz_data['toa']
            
            # 複製可變更陣列
            detection_updated = np.copy(detection)
            num_frames = end_frame - start_frame + 1
            
            # 將 t_ai 到 end_frame 範圍內的對應物件標籤改成 1
            updated_count = 0
            for f_num in range(int(t_ai), end_frame + 1):
                t = f_num - start_frame
                if 0 <= t < num_frames:
                    for obj_idx in range(30):
                        if int(detection_updated[t, obj_idx, 0]) == risky_track_id:
                            detection_updated[t, obj_idx, 5] = 1.0
                            updated_count += 1
            
            # 重新打包儲存
            np.savez_compressed(npz_path, feature=feature, detection=detection_updated, vid_id=vid_id, toa=toa)
            logger.info(f"[標籤更新成功] 已成功將 {updated_count} 個時序節點的物件 ID {risky_track_id} 風險標籤覆寫為 1.0")
        except Exception as e:
            logger.error(f"寫入 .npz 檔案發生錯誤 {npz_path}: {e}")
    else:
        if risky_track_id == 0:
            logger.info(f"該片段經判定為安全或無主要危險物件，不執行 .npz 標籤複寫。")
        else:
            logger.warning(f"找不到對應影片的實體 .npz 特徵檔案，無法回填標籤: {npz_path}")

def process_txt(path, processor, llm):
    logger.info(f"正在載入: {path}")
    df_benchmark = parse_benchmark_txt(path)
    logger.info(f"成功載入，共計 {len(df_benchmark)} 個短片片段待處理。")

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

        # 定位對應大容量磁碟上的實體 .npz 檔案路徑
        npz_path = os.path.join("/media/wayne/27CD255760735841/MMAU/", chosen_dataset, accident_type, folder_name, f"clip_{start_f}_{end_f}.npz")

        process_clip(image_dir, npz_path, video_id_str, start_f, end_f, label_val, t_ai, description, processor, llm)
        
    logger.info(f"🎉 處理完 {path} ！！\n")

if __name__ == '__main__':
    checkpoint_path = "Qwen/Qwen3-VL-8B-Instruct-FP8"
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

    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/mini_test.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/mini_training.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/mini_val.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/full_training.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/full_val.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/full_test_5s.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/full_test_4s.txt', processor, llm)
    process_txt('/home/wayne/Desktop/Thesis_code/RAM/configs/full_test_2s.txt', processor, llm)