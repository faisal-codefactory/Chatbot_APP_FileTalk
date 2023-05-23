from pytz import timezone
# import tzlocal
# import pytz
from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
import faiss
from pathlib import Path
import uvicorn
import jwt
from datetime import timedelta
import pyodbc
from datetime import datetime
import random
import string
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import configparser
import openai
from functools import wraps
import logging
from typing import List
import json
import tiktoken
import re
import uvicorn
import mimetypes
from chatbot_utils import IncomingFileProcessor
import os
import tempfile
import pickle
from elevenlabs import generate, play, voices

# from auth.auth_bearer import JWTBearer
# from auth.auth_handler import signJWT, decodeJWT

load_dotenv()
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT secret key must be set using JWT_SECRET_KEY environment variable.")
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_TIME_DAYS = 7

# Setup the custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('chat_app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
load_dotenv()
# Setup openai key
# cfg_reader = configparser.ConfigParser()
# fpath = Path.cwd() / Path('.env')
# cfg_reader.read(str(fpath))
# os.environ["OPENAI_API_KEY"] = cfg_reader.get('API_KEYS', 'OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

try:
    os.mkdir('tempfiles')
except:
    pass
# Setup parameters
system_rol_dict = {"role": "system",
                   "content": "You are a helpful Assistant. Create concise Answer with engaging tone. "}
file_processor = IncomingFileProcessor(chunk_size=250)

embed_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
# embed_model_name = 'LLukas22/all-MiniLM-L12-v2-embedding-all'
# embed_fn = HuggingFaceEmbeddings(model_name=embed_model_name)

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def get_token(request: Request):
    token = None
    auth_header = request.headers.get('Authorization')
    if auth_header:
        bearer_token = auth_header.split()
        if len(bearer_token) == 2 and bearer_token[0] == 'Bearer':
            token = bearer_token[1]
    return token


def authenticate_user(token: str = Depends(get_token)):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email: str = payload.get('sub')
        id: int = payload.get('id')
        if email is None:
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return [email, id]


class User(BaseModel):
    Email: str
    Password: str
    UserName: str


class FileModel(BaseModel):
    title: str
    description: str


class Text(BaseModel):
    text: str


class UserData(BaseModel):
    email: str
    password: str


class Database:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Get environment variables
        sql_host = os.environ.get('SQL_HOST')
        sql_UserId = os.environ.get('USERID')
        sql_passowrd = os.environ.get('PASSWORD')
        sql_database = os.environ.get('DATABASE')

        print(pyodbc.drivers())

        drivers = [item for item in pyodbc.drivers()]
        driver = drivers[-1]
        print("driver:{}".format(driver))
        server = os.environ.get('SQL_HOST')
        database = os.environ.get('DATABASE')
        uid = os.environ.get('USERID')
        pwd = os.environ.get('PASSWORD')
        con_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={uid};PWD={pwd}'
        # con_string = f'DRIVER={pyodbc.drivers()[0]};SERVER={server};DATABASE={database};'
        print(con_string)
        self.conn = self.conn = pyodbc.connect(con_string)

    async def user_exists(self, email):

        cursor = self.conn.cursor()
        cursor.execute(f"Select * from UserDetails where email = '{email}'")
        row = cursor.fetchone()
        if row:
            return 0
        else:
            return 1

    async def insert_user_details(self, user):
        try:
            result = await self.user_exists(user.Email)
            if result == 1:
                sql = "INSERT INTO UserDetails (Email, UserName, Password, RoleId,CreatedDate,isDelete) VALUES (?,?, ?,?,?,?)"
                cursor = self.conn.cursor()
                cursor.execute(sql, (user.Email, user.UserName,
                                     user.Password, 0, datetime.now(), 0))

                cursor.commit()
                cursor.close()
                return 1
            else:
                return 0
        except:
            return 0

    async def login_user(self, email, password):
        try:
            cursor = self.conn.cursor()
            sql = f"select id,UserName,email,createddate from userdetails where email = ? and password = ? "
            cursor.execute(sql, (email, password))
            row = cursor.fetchone()
            if row:
                return row
            return None

        except:
            return None

    async def get_current_bot_data(self, botid):
        cursor = self.conn.cursor()
        query = f"select id,category,description from VectorDatabase where id = '{botid}' and isdelete = 0"
        cursor.execute(query)
        row = cursor.fetchall()
        if row is None:
            return None
        data = dict()
        for r in row:
            data = {
                'id': r[0],
                'category': r[1],
                'description': r[2]
            }

        return data

    async def delete_bot(self, botid):
        cursor = self.conn.cursor()
        cursor.execute(
            f"Select IsDelete from vectordatabase where id = {botid}")
        row = cursor.fetchone()
        if row is None:
            return 0
        if row[0] == 1:
            return 2
        query = "UPDATE vectordatabase SET IsDelete = 1 where id = ?"
        cursor.execute(query, botid)
        cursor.commit()
        return 1

    async def get_bot_history(self, botid):
        cursor = self.conn.cursor()
        query = "select question,answer,DateAndTime from HistoryData where botid = ?"
        cursor.execute(query, botid)
        rows = cursor.fetchall()
        hist_list = []
        for row in rows:
            data = {
                'question': row[0],
                'answer': row[1],
                'DateAndTime': str(row[2]),

            }
            hist_list.append(data)
        return hist_list

    async def getallbots(self):
        cursor = self.conn.cursor()
        botlist = list()
        query = f"SELECT id,category,description FROM VectorDatabase where IsDelete = 0"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            data = {
                'id': row[0],
                'category': row[1],
                'description': row[2]

            }
            botlist.append(data)
        return botlist

    async def insert_into_history_table(self, botid, quest, ans, datetime):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO HistoryData(botid,question,answer,DateAndTime,isdelete) VALUES (?,?,?,?,?)",
                       (botid, quest, ans, datetime, 0))
        cursor.commit()
        return 1

    async def insert_vector_data(self, response, title, desc):
        cursor = self.conn.cursor()

        cursor.execute(
            "Insert into VectorDatabase (FilePath,category,IsDelete,Description) values (?,?, ?,?)",
            (response, title, 0, desc))
        cursor.commit()

        return 1

    async def get_vector_data(self, id):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT FilePath FROM VectorDatabase where id = {id}")
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            return None


db = Database()


async def generate_audio(text_data: str):
    text = text_data
    audio = generate(
        voice="Adam",
        model='eleven_multilingual_v1',
        text=text
    )
    # Convert audio bytes to a file-like object
    audio_file = BytesIO(audio)

    # Return the audio as a streaming response with the appropriate MIME type return StreamingResponse({"message":
    # 'File save successfully', 'statusCode': 200, 'data': response_data, 'audio': audio,}, media_type="audio/wav")

    return StreamingResponse(audio_file, media_type="audio/wav")


def openai_key_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not openai.api_key:
            raise ValueError(
                "OpenAI API key must be set using openai.api_key before calling this function.")
        try:
            openai.Model.list()
            return func(*args, **kwargs)
        except openai.error.AuthenticationError as e:
            raise ValueError("Invalid OpenAI API key. Please Replace Your Key")

    return wrapper


@openai_key_required
# def qa_chain(vectordb, chat_history, num_chunks: int):
def create_qa_chain(vectordb, num_chunks: int):
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": num_chunks})
        llm_model = ChatOpenAI(model='gpt-3.5-turbo',
                               temperature=0.6, openai_api_key=openai_api_key)
        # The difference in qa and conv_qa is the chat_history. Conv_qa maintains chat history but qa can not.
        qa = RetrievalQA.from_chain_type(
            llm=llm_model, chain_type='stuff', retriever=retriever)
        logger.info('QA_Chain created successfully')
        # conv_qa = ConversationalRetrievalChain.from_llm(
        #    llm=llm_model, chain_type='stuff', retriever=retriever, chat_history=chat_history)
        return qa
    except Exception as ex:
        logger.critical(f'QA_Chain creation Failed with error: {str(ex)}')
        raise


def run_qa_chain(qa_chain, query):
    try:
        response = qa_chain({"query": query})
        return response
    except:
        raise


@app.get("/")
async def index():
    return "Welcome to BIH Chatbot App... "


async def ingest_pdf(title: str, desc: str, file: UploadFile = File(...)):
    # userid = token[1]
    try:
        contents = await file.read()
        if contents:
            logger.info('PDF loaded successfully')
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(contents)
                pdf_file = tmp_file.name
        else:
            logger.critical('Unable to receive PDF file')
        texts = file_processor.get_pdf_splits(pdf_file)
        logger.info('PDF PDF splitting successful')
        vectordb = file_processor.create_new_vectorstore(texts, embed_fn)
        logger.info('VectorDB created successfully')
        # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        curr_date = str(datetime.now())
        filename = "".join(random.choices(string.ascii_letters, k=4)
                           ) + curr_date.split('.')[0].replace(':', '-')
        # path_file = (Path(__file__).parent).joinpath('tempfiles')
        path_file = (Path(__file__).parent).joinpath('tempfiles', filename)
        with open(path_file, 'wb') as tmp_file:
            # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(vectordb, tmp_file)
            tmp_path = tmp_file.name

        logger.info('VectorDB Prepared for Transfer')
        response = await db.insert_vector_data(tmp_path, title, desc)
        # return StreamingResponse(open(tmp_path, "rb"), media_type="application/octet-stream")
        return 1

    except Exception as ex:
        logger.critical(f'unable to create vectordb: {str(ex)}')
        response_data = {
            'Unsuccessful': f'VectorDB Creation Unsuccessful, {str(ex)}'}
        return JSONResponse(response_data)


async def ingest_docx(title: str, desc: str, file: UploadFile = File(...)):
    try:
        # userid = token[1]
        contents = await file.read()
        if contents:
            logger.info('Docx loaded successfully')
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(contents)
                docx_file = tmp_file.name
        else:
            logger.critical('Unable to receive Docx file')
        texts = file_processor.get_docx_splits(docx_file)
        logger.info('Docx splitting successful')
        vectordb = file_processor.create_new_vectorstore(texts, embed_fn)
        logger.info('VectorDB created successfully')
        curr_date = str(datetime.now())
        filename = "".join(random.choices(string.ascii_letters, k=4)
                           ) + curr_date.split('.')[0].replace(':', '-')
        path_file = (Path(__file__).parent).joinpath('tempfiles', filename)
        with open(path_file, 'wb') as tmp_file:
            # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(vectordb, tmp_file)
            tmp_path = tmp_file.name

        logger.info('VectorDB Prepared for Transfer')
        response = await db.insert_vector_data(tmp_path, title, desc)
        return 1

        # return StreamingResponse(open(tmp_path, "rb"), media_type="application/octet-stream", headers={
        # "Content-Disposition": "attachment;filename=vectordb.pkl"}) return StreamingResponse(open(tmp_path, "rb"),
        # media_type="application/octet-stream")
    except Exception as ex:
        logger.critical(f'unable to create vectordb: {str(ex)}')
        response_data = {
            'Unsuccessful': f'VectorDB Creation Unsuccessful, {str(ex)}'}
        return JSONResponse(response_data)


async def ingest_zip(title: str, desc: str, file: UploadFile = File(...)):
    try:
        # userid = token[1]
        contents = await file.read()
        if contents:
            logger.info('Zip loaded successfully')
            zip_stream = BytesIO(contents)
            # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            #    tmp_file.write(contents)
            #    pdf_file = tmp_file.name
        else:
            logger.critical('Unable to receive Zip file')

        texts = file_processor.get_zip_splits(zip_stream)
        assert (texts is not None)
        logger.info('Zip splitting successful')

        # Vectordb preparation
        vectordb = file_processor.create_new_vectorstore(texts, embed_fn)
        logger.info('VectorDB created successfully')
        curr_date = str(datetime.now())
        filename = "".join(random.choices(string.ascii_letters, k=4)
                           ) + curr_date.split('.')[0].replace(':', '-')
        path_file = (Path(__file__).parent).joinpath('tempfiles', filename)
        with open(path_file, 'wb') as tmp_file:
            # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(vectordb, tmp_file)
            tmp_path = tmp_file.name
        logger.info('VectorDB Prepared for Transfer')

        response = await db.insert_vector_data(tmp_path, title, desc)
        return 1
        # return StreamingResponse(open(tmp_path, "rb"), media_type="application/octet-stream")
        # return JSONResponse({"message": 'File save successfully', 'statusCode': 200})

    except Exception as ex:
        logger.critical(f'unable to create vectordb: {str(ex)}')
        response_data = {
            'Unsuccessful': f'VectorDB Creation Unsuccessful, {str(ex)}'}
        return JSONResponse(response_data)


@app.post("/bot/ingest_file_to_db")
async def ingestall_data(title: str, description: str, file: UploadFile = File(...)):
    if title is None:
        return JSONResponse({"message": 'Please enter a title', 'statusCode': 204})
    if file is None:
        return JSONResponse({"message": 'Please Select a File', 'statusCode': 204})
    response = None
    if '.zip' in file.filename[-4:] or '.ZIP' in file.filename[-4:]:
        response = await ingest_zip(title, description, file)
    elif '.pdf' in file.filename[-4:] or '.PDF' in file.filename[-4:]:
        response = await ingest_pdf(title, description, file)
    elif '.docx' in file.filename[-5:] or '.DOCX' in file.filename[-4:]:
        response = await ingest_docx(title, description, file)
    if response == 1:
        return JSONResponse({"message": 'File save successfully', 'statusCode': 200, 'data': []})
    else:
        return JSONResponse({"message": 'Please Enter a PDF/DOCX/ZIP file', 'statusCode': 422, 'response': response})


@app.get("/bot/get_LLM_response_from_query_with_vectordb/{botid}")
async def get_llm_response_from_vectordb(query: str, botid: int):
    try:
        response = await db.get_vector_data(botid)
        if response is None:
            return JSONResponse({"statusCode": 404, "message": "Select a Valid BOT"})
        vectordb = pickle.loads(open(response, "rb").read())
        # contents = await vector_file.read()
        # if contents:
        #     vectordb = pickle.loads(contents)
        #     logger.info('VectorDB loaded and de-serialized successfully')
        # else:
        #     logger.critical('Unable to load/deserialize VectorDB')
    except Exception as ex:
        logger.critical(f'Failure: {str(ex)}')
        response_data = {'Unsuccessful': json.dumps(
            f'VectorDB Loading Unsuccessful,{str(ex)}')}
        return JSONResponse(response_data)
    # Create qa_chain and get response
    try:
        qa_chain = create_qa_chain(vectordb, 6)
        res = run_qa_chain(qa_chain, query)

        # audio = await generate_audio(str(res['result']))
        current_time = datetime.now()
        history_response = await db.insert_into_history_table(botid, query, json.dumps(str(res['result'])),
                                                              current_time)
        response_data = {'Answer:': str(res['result']), 'Date': str(current_time).split()[0],
                         'Time': str(current_time).split()[1].split('.')[0]}
        # return JSONResponse({ 'statusCode': 200, 'data': response_data,'audio':audio})
        return JSONResponse({'statusCode': 200, 'DateAndTime': str(current_time), 'data': str(res['result'])})

    except Exception as ex:
        logger.critical(
            f'query answer failed due to chain creation: {str(ex)}')
        response_data = {
            'Unsuccessful': f'Chain Creation/Response Generation Unsuccessful, {str(ex)}'}
        return JSONResponse({"statusCode": 500, "data": response_data})


@app.get('/bot/getposts')
async def get_all_posts(request: Request):
    response = await db.getallbots()
    if response is not None:
        return JSONResponse({'message': 'Retreive data', 'data': response, 'statusCode': 200})
    return JSONResponse({"message": 'Not found', 'statusCode': 404, 'data': []})


@app.get("/bot/gethistory/{botid}")
async def get_bot_history_data(botid):
    response = await db.get_bot_history(botid)
    if response:
        return JSONResponse({'Data': response, 'statusCode': 200})
    return JSONResponse({'message': 'Not Found', 'statusCode': 404, 'data': []})


@app.delete('/bot/deletebot/{botid}')
async def delete_current_bot(botid: int):
    if botid == 1:
        return JSONResponse({"message": 'Can not delete this bot', 'statusCode': 405, 'data': []})
    response = await db.delete_bot(botid)
    if response == 1:
        return JSONResponse({"message": 'Bot Deleted', 'statusCode': 200, 'data': []})
    elif response == 0:
        return JSONResponse({"message": 'Select a Valid BOT', 'statusCode': 404, 'data': []})
    elif response == 2:
        return JSONResponse({"message": 'BOT is already deleted', 'httpStatusCode': 404})
    return JSONResponse({"message": 'Failed!!', 'statusCode': 500, 'data': []})


@app.get('/bot/getbotdata/{botid}')
async def get_bot_id(botid: int):
    response = await db.get_current_bot_data(botid)
    if response:
        return JSONResponse({'statusCode': 200, 'data': response})
    return JSONResponse({'message': 'Not Found', 'statusCode': 404, 'data': []})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7070)
