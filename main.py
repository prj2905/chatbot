from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import get_bot_response
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://fund-spark-bu2i-jall49o1q-prj2905s-projects.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    query: str

@app.post("/chat/")
def chat(msg: Message):
    reply = get_bot_response(msg.query)
    return {"response": reply}
