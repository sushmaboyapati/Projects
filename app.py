import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI(title="AI Chatbot API")

# Request schema
class ChatRequest(BaseModel):
    user_input: str

# Response schema
class ChatResponse(BaseModel):
    response: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "AI Chatbot is running!"}

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # lightweight model
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": request.user_input}
            ],
            temperature=0.7,
            max_tokens=200
        )

        reply = completion.choices[0].message.content

        return ChatResponse(response=reply)

    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

