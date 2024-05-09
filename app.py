
from fastapi import FastAPI
import uvicorn
from transformers import pipeline

import os
from dotenv import load_dotenv
load_dotenv()
p = 8080
# line no 6,7,8,9 helps to get the variable PORT from the .env file

import pickle
with open('HF_summarizer.pkl', 'rb') as f:
    HF_summarizer = pickle.load(f)


app = FastAPI()


@app.get('/')
def welcome():
    return ({"messages" : "Hello World !!!!"})

@app.post('/infer')
def get_inference(data:str):

    answer = HF_summarizer(data)
    return answer[0]['summary_text']


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = p)

### Note that it took me about 3 hours to figure out that the model MUST be in 
### pickle format if we want to write code on .py file using a summarizer model
### pipe = pipeline(task = "summarization")
### pipe(any_string_data) will work in .ipynb
### But to use the model as an API, it has to be in a pickle file









