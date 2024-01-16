# Import libraries
from fastapi import FastAPI

#from fastapi import FastAPI, HTTPException
#from src.pred.image_classifier import *
#from fastapi.middleware.cors import CORSMiddleware
#from src.schemas.image_schema import Img


from fastapi import FastAPI
from pydantic import BaseModel

# Define a request model
class InferenceRequest(BaseModel):
    data: str  # Replace 'str' with the data type you expect

# Initialize the FastAPI app
app = FastAPI()

# Define an endpoint for inference
@app.post("/infer/")
async def infer(request: InferenceRequest):
    # Replace the next line with your model's inference logic
    response = f"Received data: {request.data}"

    return {"response": response}

# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


