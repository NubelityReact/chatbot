from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser   
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from typing import Literal
from .templates import rag_template, system_template
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os


class ConversationItem(BaseModel):
    agent: Literal["user", "ai"]
    message: str

API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()

@router.post("/chatbot/")
def responseQuery(Conversation: list[ConversationItem]): 
    try:
        conversation_messages = [HumanMessage(content=item.message, name=item.agent) if item.agent == "user" else AIMessage(name=item.agent, content=item.message)  for item in Conversation[:-1]]
        messages = [SystemMessage(name="system",content=system_template())] + conversation_messages 
        dynamic_prompt = ChatPromptTemplate.from_messages(messages)
        conversation_messages = dynamic_prompt.messages
        conversation_text = ""
        for item in conversation_messages:
            conversation_text = conversation_text + f"role: {item.name}\n message: {item.content}\n\n"

        prompt_template = ChatPromptTemplate.from_template(rag_template)

        # with open("db4.txt") as db:
            # doc = db.read()

        
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=55, chunk_overlap=4)
        # splits = text_splitter.split_text(doc)
        # embeddings_model = OpenAIEmbeddings(api_key=API_KEY)
        # vector_store = Chroma.from_texts(texts=splits, embedding=embeddings_model)
        # retriever = vector_store.as_retriever()

        llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o")
        output_parser = StrOutputParser()
        user_question = Conversation[-1].message
        
        
        rag_chain = (
            {"conversation_text": lambda x: conversation_text, "question": RunnablePassthrough() }
            | prompt_template
            | llm
            | output_parser
        )
        response = rag_chain.invoke(user_question)
        return response

    except Exception as e:
        print("An unexpected exception occurred: " + str(e))
        return JSONResponse(status_code=500, content={"message": "An unexpected error occurred", "error": str(e)})


@router.get("/read")
def readDB():
    with open("db.txt", "rb") as db:
        lines = db.readlines()

    messages = [SystemMessage(name="system", content="Eres un asistente de inteligencia artificial para servir y responder de forma amable, claro y conciso a lo que se te pregunta. Tu nombre es Nuby")] +[HumanMessage(name="user", content="What is string theory")] 
    print(messages)
    dynamic_prompt = ChatPromptTemplate.from_messages(messages)
    converstion_text = dynamic_prompt.messages
    print(converstion_text)
    print([(item.name, item.content) for item in converstion_text])


# Email request model
class EmailRequest(BaseModel):
    recipient: EmailStr
    subject: str
    message: str


# Utility function to send email
def send_email(recipient: str, subject: str, message: str):
    try:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = os.getenv("SMTP_PORT", 587)
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        print(smtp_server, smtp_user, smtp_password)
        
        if not smtp_user or not smtp_password:
            raise ValueError("SMTP credentials are not set in the environment variables.")

        # Create email
        email = MIMEMultipart()
        email["From"] = smtp_user
        email["To"] = recipient
        email["Subject"] = subject
        email.attach(MIMEText(message, "plain"))

        # Connect to the server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, email.as_string())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")


# Endpoint to send email
@router.post("/send-email/")
def send_email_endpoint(email_request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Send an email to the specified recipient.
    """
    background_tasks.add_task(
        send_email,
        recipient=email_request.recipient,
        subject=email_request.subject,
        message=email_request.message,
    )
    return {"message": "Email has been sent successfully."}