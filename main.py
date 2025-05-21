from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from infer import Predictor
from fastapi.responses import JSONResponse, StreamingResponse
import io
import cv2
import numpy as np
predictor = Predictor()
app = FastAPI()


@app.post("/predict/")
async def predict( file: UploadFile = File('data_source/200505_Bellingham_Mavic_Mini_VIS_0015_00000060.jpg')):
    contents = await file.read()
    image_file = io.BytesIO(contents)
    prediction = predictor.predict(image_file)
    
    _, encoded_img = cv2.imencode(".jpg", prediction)
    image_bytes = io.BytesIO(encoded_img.tobytes())
    return StreamingResponse(image_bytes, media_type="image/png")


    