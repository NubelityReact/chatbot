from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser   
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from typing import Literal


class ConversationItem(BaseModel):
    agent: Literal["user", "ai"]
    message: str

API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()

@router.post("/chatbot/")
def responseQuery(Conversation: list[ConversationItem]): 
    try:
        conversation_messages = [HumanMessage(content=item.message, name=item.agent) if item.agent == "user" else AIMessage(name=item.agent, content=item.message)  for item in Conversation[:-1]]
        messages = [SystemMessage(name="system",content="Tu nombre es Nuby. Una IA que ayuda a responder las dudas del usuario de manera clara y conscisa.")] + conversation_messages 
        dynamic_prompt = ChatPromptTemplate.from_messages(messages)
        conversation_messages = dynamic_prompt.messages
        conversation_text = ""
        for item in conversation_messages:
            conversation_text = conversation_text + f"role: {item.name}\n message: {item.content}\n\n"
        print(conversation_text)
        template = """
            Esta es la conversación hasta ahora: {conversation_text}.

            Contesta la siguiente pregunta: {question}
            Con la siguiente información como contexto: {ctx}.
            No es necesario que te presentes en cada respuesta.
            Tu respuesta debe hablar por parte de Nubelity.
            No repitas información previamente otorgada en la conversación hasta ahora, al menos que sea para recordarle al usuario con quién debería ponerse en contacto para ayuda personalizada.
            Nunca los invites directamente a visitar la siguiente homepage: www.nubelity.com
            
            Sólo si no es posible contestar a la pregunta con la información del contexto entonces 
            invítalo a llenar algunos de los formularios disponibles en la página.
        """
            # Si el usuario está interesado en:
            # 1. expasión de talento pon un link a la siguiente página: www.nubelity.com/talent-expansion
            # 2. fábrica de software pon un link a la siguiente página: www.nubelity.com/software-factory
            # 3. centro de entrenamiento pon un link a la siguiente página: www.nubelity.com/training-center
            # 4. servicios cloud pon un link a la siguiente página: www.nubelity.com/cloud-services

        prompt_template = ChatPromptTemplate.from_template(template)

        with open("db4.txt") as db:
            doc = db.read()

        # template_improve_context = """
        #     Reescribe la siguiente información: {info}
        #     de una forma mas descriptiva y detallada. El output debe contener la pregunta original: {question} y después la información que tu creaste.
        # """

        # prompt_improve_ctx = ChatPromptTemplate.from_template(template_improve_context)


        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=55, chunk_overlap=4)
        splits = text_splitter.split_text(doc)
        embeddings_model = OpenAIEmbeddings(api_key=API_KEY)
        vector_store = Chroma.from_texts(texts=splits, embedding=embeddings_model)
        retriever = vector_store.as_retriever()



        llm = ChatOpenAI(api_key=API_KEY)
        output_parser = StrOutputParser()
        # chain = prompt | llm | output_parser
        user_question = Conversation[-1].message
        
        print(user_question)
        
        # improve_db_chain = (
        #     {"info": retriever, "question": RunnablePassthrough()}
        #     | prompt_improve_ctx
        #     | llm
        #     | output_parser
        # )

        # improved_ctx = improve_db_chain.invoke(user_question)
        # print("improved ctx", improved_ctx)
        rag_chain = (
            {"conversation_text": lambda x: conversation_text, "ctx": retriever, "question": RunnablePassthrough() }
            | prompt_template
            | llm
            | output_parser
        )
        response = rag_chain.invoke(user_question)
        print(response)
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