"""
NOTE--->  Feel free to test this for a single saved model_version ONLY. Used postman for POST the image file uploading and receiving links
tutuorial :----> https://youtu.be/t6NI0u_lgNo?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD @codebasics

"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app1 = FastAPI()

MODEL = tf.keras.models.load_model('../saved_models/2')
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
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence_score = np.max(predictions[0])
    return{
        'class' : predicted_class,
        'score' : float(confidence_score)
    }


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
----> 

"""

if __name__ == "__main__":
    uvicorn.run(app1, host='localhost', port=8000)
