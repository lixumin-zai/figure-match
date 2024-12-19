import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
from io import BytesIO
import os
from matchor import Matchor

def feishu_get_feature():
    # https://vsxa1w87lf.feishu.cn/sheets/YBEDsp6YwhddwbtY8NJc0xBCnuc?sheet=2fc440
    app_id, app_key = "cli_a621015572aa100c", "iYdkBItcLDdwD90ihVGO5gxjDDpURX3b"
    sheet_token, sheet_id = "YBEDsp6YwhddwbtY8NJc0xBCnuc", "pqKPD8"
    feishu_sdk.login(app_id, app_key)
    sheet = FeishuSheet(sheet_token, sheet_id)
    matchor = Matchor()

    idx = 2
    image_col = "B"
    result_col = "O"
    data_col = "P"
    for i in range(min(sheet.rows+1, 10000)):
        print(i)
        if i < idx:
            continue 
        # if sheet[f"{result_col}{i}"]:
        #     continue
        image_bytes = sheet[f"{image_col}{i}"].image_bytes
        match_info = matchor.match_top5(image_bytes)
        image_name = match_info[0][1]
        with open(f"./database/{image_name}", "rb") as f:
            image_bytes = f.read()

        sheet[f"{result_col}{i}"] = FeishuImage(img_bytes=image_bytes)
        sheet[f"{data_col}{i}"] = str(match_info)
        
        
if __name__ == "__main__":
    feishu_get_feature()