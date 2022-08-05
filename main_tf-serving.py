from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# to make a http call
import requests

app1 = FastAPI()

endpoint1 = "http://localhost:8502/v1/models/potatoes_model:predict"
"""
NOTE:- in above endpoint_url /potatoes_model name was there occuring in the powershell by my docker image container. DO NOT give your personal url here. BEWARE !!
It can give Internal server error...
"""

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app1.get('/start')
async def start():
    return 'gg, you are there'


"""
user will upload images on the web-app, so creating a form named as File which accepts an image
// tested the api on Postman
"""

"""
@app1.post("/predict")
async def predict(
        file: UploadFile = File(...)   # here UploadFile is a type object and its value is by default File
):
    pass
"""


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app1.post("/predict")
async def predict(
        file: UploadFile = File(...)  # here UploadFile is a type object and its value is by default File
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint1, json=json_data)

    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence_score = np.max(prediction)

    return{
        "class":predicted_class,
        "confidence": confidence_score
    }

# json_Data --> it is the actual request

"""
    json_data is a dictionary where "instances"--> a key which maps image_batch(in list form)
    and, in response to this call we receive at::---
    response ----> stores the request made at the endpoint
"""

"""
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence_score = np.max(predictions[0])
    return{
        'class' : predicted_class,
        'score' : float(confidence_score)
    }
"""

"""
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    pass
"""

"""
bytes = await file.read()    # reading the file in async - await route

return
"""

"""
// ---> convert the image into a numpy tensor
---> first store the uploaded_images in byte  ---> then convert the data which is in bytes into numpy array. 
      Use PIL module and use BytesIO
---> create a Global variable named MODEL, and a CLASS_name var and now we will predict on the accepted image. NO
---> note, image does not accept a single image rather a batch of images, so we expand the dim_numpy array using 
     [256,256,3] ---> [[256,256,3]] // np.expand_dims(img,0)
----> // predictions variable is an array of confidence score for 3 classes
----> return the max_score from the array
----> NOTE:- np.argmax([1,1.1,2.1] returns index 2  | np.max([1,1.1,2.1]) returns 2.1

----> In industries, you will not rely on single model_version. 
       To dynamically load and deploy multiple model_versions use TF-serving module
       Instead of changing the code every-time for altering the model_version (very Dangerous),
       we TWEAK into .config file  

"""

if __name__ == "__main__":
    uvicorn.run(app1, host='localhost', port=8000)
