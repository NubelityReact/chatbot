from fastapi import FastAPI
from .routers import chatbot
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from starlette.middleware.cors import CORSMiddleware


load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(chatbot.router)

@app.get("/")
def intial():
    return "Hello world from fastapi 8000"


@app.get("/favicon.ico")
def favicon():
    return FileResponse("./favicon.ico")

