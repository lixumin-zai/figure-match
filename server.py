from fastapi import FastAPI
from starlette.responses import StreamingResponse
import time
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
from matchor import Matchor
from multiprocessing import Pool, cpu_count

app = FastAPI()

matchor = Matchor()

# 定义请求数据的模型
class UploadFile(BaseModel):
    image: str

@app.post("/search")
async def stream(upload_file:UploadFile):
    image_bytes = base64.b64decode(upload_file.image)
    match_image_bytes = matchor(image_bytes)
    
    #return {"code":"0", "image": base64.b64encode(match_image_bytes).decode("utf-8")}
    return {"code":"0", "image": [base64.b64encode(i).decode("utf-8") for i in match_image_bytes]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=20010)
