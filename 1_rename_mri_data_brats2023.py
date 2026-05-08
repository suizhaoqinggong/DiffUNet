



import os 

data_dir = "/home/cjh/data/BraTS2021/"
# data_dir = "/data/chenjiahao/raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"

all_cases = os.listdir(data_dir)

for case_name in all_cases:
    case_dir = os.path.join(data_dir, case_name)

    if not os.path.isdir(case_dir):
        continue

    for data_name in os.listdir(case_dir):

        if "-" not in data_name:
            continue
        new_name = data_name.split("-")[-1]

        new_path = os.path.join(case_dir, new_name)
        old_path = os.path.join(case_dir, data_name)

        # 如果目标文件已存在，跳过
        if os.path.exists(new_path):
            print(f"跳过 {data_name}，目标文件已存在")
            continue

        os.rename(old_path, new_path)

        print(f"{new_path} 命名成功")

