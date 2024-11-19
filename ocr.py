from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model('ocrmodel.keras')

class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None, "image": None})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = Image.open(BytesIO(image_data))


        img = img.convert('RGB')  
        img = img.resize((64, 64))  
        img_array = np.array(img)  
        img_array = img_array / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

    
        predictions = model.predict(img_array)


        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        
  
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

       
        prediction_result = {
            "class_label": predicted_class_label,
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": prediction_result,
            "image": f"data:image/jpeg;base64,{img_base64}"  
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": f"Error: {str(e)}",
            "image": None
        })
