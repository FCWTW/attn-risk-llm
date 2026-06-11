import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

ROOT_DIR = "/home/wayne/Documents/MMAU"
model = YOLO("yolo11s.pt")

VALID_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle"
}

def parse_benchmark_txt(txt_path):
    """
    解析 Benchmark 的 .txt 檔案，轉換為 DataFrame 導航表
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
                toa_clean = parts[4].split(',')[0]
                parsed_data.append([
                    parts[0],       # video_id (例如 "1/001537" 或 "1/001")
                    int(parts[1]),  # label 
                    int(parts[2]),  # start_frame
                    int(parts[3]),  # end_frame
                    float(toa_clean)# toa
                ])
                
    return pd.DataFrame(
        parsed_data, 
        columns=['video_id', 'label', 'start_frame', 'end_frame', 'toa']
    )

def process_clip(image_dir, start_frame, end_frame):
    """
    針對特定時間切片內的影像進行 YOLO11s + ByteTrack 連續追蹤
    """
    all_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # 1. 根據開頭與結尾影格，過濾出該 Clip 所需的影像與 ID
    clip_image_paths = []
    clip_frame_ids = []
    
    for fname in all_files:
        try:
            # 轉換為整數（完美相容 000001.jpg 與 0001.png 的零填充）
            frame_num = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
            
        if start_frame <= frame_num <= end_frame:
            clip_image_paths.append(os.path.join(image_dir, fname))
            clip_frame_ids.append(os.path.splitext(fname)[0])

    if not clip_image_paths:
        return None

    records = []

    # 2. 逐影格流式追蹤核心 (串聯動態軌跡並定時重設)
    for idx, (frame_id, img_path) in enumerate(zip(clip_frame_ids, clip_image_paths)):
        
        # 核心優化：如果是 Clip 的第一張影格，設定 persist=False 來強制清空上一個短片的記憶
        # 隨後的影格設定 persist=True，維持時序連續追蹤
        is_first_frame = (idx == 0)
        
        results = model.track(
            source=img_path,
            tracker="bytetrack.yaml",
            persist=not is_first_frame,  
            verbose=False
        )

        result = results[0]
        if result.boxes is None:
            continue
            
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]

            if cls_name not in VALID_CLASSES:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 取得 ByteTrack 分配的唯一 ID
            track_id = int(box.id.item()) if box.id is not None else -1

            records.append([
                frame_id,
                track_id,
                cls_name,
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                conf
            ])

    return pd.DataFrame(
        records,
        columns=[
            "frame",
            "track_id",
            "label",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence"
        ]
    )

def process_txt(path):
    print(f"正在載入基準測試導航表: {path}")
    df_benchmark = parse_benchmark_txt(path)
    print(f"成功載入，共計 {len(df_benchmark)} 個短片片段 (Clips) 待處理。")

    for idx, row in tqdm(df_benchmark.iterrows(), total=len(df_benchmark), desc="總進度"):
        video_id_str = row['video_id']  
        start_f = row['start_frame']
        end_f = row['end_frame']
        
        accident_type, folder_name = video_id_str.split('/')
        
        # 自動適應 CAP-DATA 與 DADA-DATA 的目錄佈局
        chosen_dataset = None
        for dataset_name in ["CAP-DATA", "DADA-DATA"]:
            potential_dir = os.path.join(ROOT_DIR, dataset_name, accident_type, folder_name, "images")
            if os.path.exists(potential_dir):
                chosen_dataset = dataset_name
                break
                
        if chosen_dataset is None:
            print(f"\n[警告] 根據結構找不到該影片實體路徑: {video_id_str}，已跳過。")
            continue
            
        video_root_dir = os.path.join(ROOT_DIR, chosen_dataset, accident_type, folder_name)
        image_dir = os.path.join(video_root_dir, "images")

        df_clip_res = process_clip(image_dir, start_f, end_f)

        if df_clip_res is not None:
            output_csv = os.path.join(
                video_root_dir,
                f"detections_{start_f}_{end_f}.csv"
            )
            df_clip_res.to_csv(output_csv, index=False)
            
    print(f"🎉 處理完 {path} ！！")

if __name__ == "__main__":
    process_txt('/home/wayne/Documents/Progress/RAM/configs/mini_training.txt')
    process_txt('/home/wayne/Documents/Progress/RAM/configs/mini_test.txt')
    process_txt('/home/wayne/Documents/Progress/RAM/configs/full_training.txt')
    process_txt('/home/wayne/Documents/Progress/RAM/configs/full_test_5s.txt')
    process_txt('/home/wayne/Documents/Progress/RAM/configs/full_test_4s.txt')
    process_txt('/home/wayne/Documents/Progress/RAM/configs/full_test_2s.txt')