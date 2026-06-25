import torch
print(torch.__version__)            # 檢查 PyTorch 版本
print(torch.cuda.is_available())    # 應該回傳 True
print(torch.cuda.device_count())    # 檢查可用的 GPU 數量
print(torch.cuda.get_device_name(0)) # 顯示第一個 GPU 的名稱

