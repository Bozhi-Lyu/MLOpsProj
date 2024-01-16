from fastapi import FastAPI, File, UploadFile, HTTPException
from src.models.model import DeiTClassifier
import torch
import uvicorn

# Assumption is a tensor input

app = FastAPI()

# Load the trained model
model = DeiTClassifier()
model.load_state_dict(torch.load('models/saved_models/model.pt', 
                                 map_location=torch.device('cpu'))) # From train model py
model.eval()

@app.post("/predict/")
async def make_prediction(file: UploadFile = File(...)):
    try:

        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            # Convert outputs to probabilities and then to a list
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            # Convert predictions to a Python list and return
            return {
                "class_id": predictions.item(),
                "confidence": confidences.item()
            }
    # If an error occurs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 
    
#if __name__ == "__main__": # If the script is run directly
#    uvicorn.run(app, host="localhost", port=8002) # Run the FastAPI app using uvicorn