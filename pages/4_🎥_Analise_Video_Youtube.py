# /pages/1_🎥_Analise_Video_Youtube.py (VERSÃO FINAL DE PRODUÇÃO)

import os
import streamlit as st
from io import BytesIO
import re

from pytube import YouTube
import openai

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


st.title("🎥 Oráculo - Análise de Vídeo do YouTube")
st.caption("Cole a URL, escolha o modelo de análise e extraia insights poderosos!")

CONFIG_MODELOS_ANALISE = {
    'Groq': {'modelos': ['Llama3.1-405b-reasoning', 'llama3-70b-8192'], 'api_key': os.getenv("GROQ_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
}

def is_valid_youtube_url(url):
    if not url: return False
    youtube_regex = re.compile(r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}.*')
    return youtube_regex.match(url) is not None

@st.cache_data(show_spinner="Buscando e transcrevendo áudio com a API Whisper...")
def transcrever_video(url_video):
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("Chave da API da OpenAI não encontrada nos Secrets do Streamlit!")
            return None

        yt = YouTube(url_video)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        buffer = BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        buffer.name = 'audio.mp3'
        
        client = openai.OpenAI(api_key=openai_api_key)
        transcricao = client.audio.transcriptions.create(model="whisper-1", file=buffer, response_format="text")
        return transcricao
    except Exception as e:
        st.error(f"Falha ao processar o vídeo. Verifique se o vídeo é público e a URL está correta. Erro técnico: {e}")
        return None

def inicializar_analise(provedor, modelo, url_video):
    texto_transcrito = transcrever_video(url_video)
    if not texto_transcrito: return

    with st.spinner("Vetorizando o texto transcrito..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([texto_transcrito])
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever_video = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    mapa_chat = {'Groq': ChatGroq, 'Gemini': ChatGoogleGenerativeAI}
    config_provedor = CONFIG_MODELOS_ANALISE[provedor]
    api_key_analise = config_provedor['api_key']
    if not api_key_analise:
        st.error(f"Chave da API para {provedor} não encontrada nos Secrets!")
        return
        
    ChatModelClass = mapa_chat[provedor]
    if provedor == 'Gemini':
        chat_model = ChatModelClass(model=modelo, google_api_key=api_key_analise)
    else:
        chat_model = ChatModelClass(model=modelo, api_key=api_key_analise)
        
    st.session_state.chat_model_video = chat_model
    st.success("Análise pronta! Pode fazer suas perguntas.")

# --- Interface e Lógica do Chat (sem alterações) ---
# ... (o resto do código permanece o mesmo) ...