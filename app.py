from fastapi import FastAPI, File, UploadFile, HTTPException
from src.models.model import DeiTClassifier
import torchvision.transforms as transforms

import torch
import uvicorn
import PIL

# Assumption is a tensor input

app = FastAPI()
model_checkpoint = "models/saved_models/model.pt"
# Load the trained model
model = DeiTClassifier()
model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))

model.eval()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_tensor/")
async def predict_tensor(file:UploadFile = File(...)):
    # read tensor pt file
    try:
        with open('tensor.pt', 'wb') as tensor:
            content = await file.read()
            tensor.write(content)
            tensor.close()
        tensor = torch.load('tensor.pt')
        with torch.no_grad():
            outputs = model(tensor)
            # Convert outputs to probabilities and then to a list
            confidences, predictions = torch.max(outputs, 1)
            print(predictions.tolist())
            # Convert predictions to a Python list and return
            return {
                "class_id": predictions.tolist(),
                #"confidence": list(confidences.numpy())
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 


# do the same but for single image 

@app.post("/predict_image/")
async def predict_image(file:UploadFile = File(...)):
    # read image file
    try:
        with open('image.jpg', 'wb') as image:
            content = await file.read()
            image.write(content)
            image.close()
        image = PIL.Image.open('image.jpg')

        # transform image to 48x48 tensor with batch size = 1
        transform = transforms.Compose(
            [
                transforms.Resize((48, 48), antialias=True),  # Zoom in on the image
                transforms.ToTensor(),  # Convert to a tensor
                # rescale to 0-1
            ]
        )

        image = transform(image) / 255.0
        
        image = image.unsqueeze(0)


        with torch.no_grad():
            outputs = model(image)
            # Convert outputs to probabilities and then to a list
            confidences, predictions = torch.max(outputs, 1)
            # Convert predictions to a Python list and return
            return {
                "class_id": predictions.tolist(),
                #"confidence": list(confidences.numpy())
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

