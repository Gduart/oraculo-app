# /pages/1_🎥_Analise_Video_Youtube.py

import os
import streamlit as st
from dotenv import load_dotenv
import time
from io import BytesIO
import re  # Biblioteca de expressões regulares para validação

# --- Importações Robustas ---
from pytube import YouTube
import openai

# --- Importações do LangChain ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

load_dotenv()

# --- Configurações Iniciais ---
st.set_page_config(page_title="Análise de Vídeo", page_icon="🎥", layout="wide")
st.title("🎥 Oráculo - Análise de Vídeo do YouTube")
st.caption("Cole a URL de um vídeo específico do YouTube, configure o modelo e extraia insights!")

CONFIG_MODELOS = {
    'Groq': {'modelos': ['Llama3.1-405b-reasoning', 'llama3-70b-8192'], 'api_key': os.getenv("GROQ_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
}

# --- Funções de Lógica ---

def is_valid_youtube_url(url):
    """Verifica se a URL fornecida é um link válido de um vídeo específico do YouTube."""
    if not url:
        return False
    # <-- CORREÇÃO: Usando 'raw string' (r'...') para evitar SyntaxWarning -->
    youtube_regex = re.compile(
        r'^(https?://)?(www\.)?'
        r'(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)'
        r'([\w-]{11})$')
    
    return youtube_regex.match(url) is not None

@st.cache_data(show_spinner="Buscando e transcrevendo o áudio do vídeo... (isso pode levar alguns minutos)")
def transcrever_via_api(url_video):
    """Usa Pytube e a API Whisper para transcrever o áudio do vídeo."""
    try:
        yt = YouTube(url_video)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        buffer = BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        buffer.name = 'audio.mp3'
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        transcricao = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            response_format="text"
        )
        return transcricao
    except Exception as e:
        st.error(f"Falha ao processar o vídeo. Verifique se o vídeo é público e a URL está correta. Erro: {e}")
        return None

def inicializar_analise_video(provedor, modelo, url_video):
    texto_transcrito = transcrever_via_api(url_video)
    
    if not texto_transcrito:
        st.warning("A transcrição falhou. Não foi possível continuar.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.create_documents([texto_transcrito])
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.retriever_video = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    mapa_chat = {'Groq': ChatGroq, 'Gemini': ChatGoogleGenerativeAI}
    api_key = CONFIG_MODELOS[provedor]['api_key']
    ChatModelClass = mapa_chat[provedor]
    
    if provedor == 'Gemini':
        chat_model = ChatModelClass(model=modelo, google_api_key=api_key)
    else:
        chat_model = ChatModelClass(model=modelo, api_key=api_key)
        
    st.session_state.chat_model_video = chat_model
    st.success("Análise de vídeo pronta! Pode fazer suas perguntas.")

# --- Interface do Usuário ---

if "memoria_video" not in st.session_state:
    st.session_state.memoria_video = []

with st.sidebar:
    st.header("🛠️ Configurar Análise de Vídeo")
    url_video = st.text_input("Cole a URL do vídeo do YouTube aqui", key="url_video_input")
    
    provedor = st.selectbox("Provedor de Análise", list(CONFIG_MODELOS.keys()), key="provedor_video")
    if provedor:
        modelo = st.selectbox("Modelo de Análise", CONFIG_MODELOS[provedor]['modelos'], key="modelo_video")
            
    if st.button("🚀 Iniciar Análise do Vídeo", use_container_width=True, type="primary"):
        # <-- CORREÇÃO: Lógica de validação mais clara -->
        if is_valid_youtube_url(url_video):
            if provedor and 'modelo' in locals():
                st.session_state.memoria_video = []
                inicializar_analise_video(provedor, modelo, url_video)
            else:
                st.warning("Por favor, selecione um provedor e um modelo.")
        else:
            st.error("URL inválida! Por favor, insira o link completo de um vídeo específico do YouTube (ex: https://www.youtube.com/watch?v=...).")
            
    if st.session_state.get('memoria_video'):
        if st.button("🗑️ Apagar Histórico", use_container_width=True):
            st.session_state.memoria_video = []
            st.rerun()

# --- Lógica do Chat Principal (sem alterações) ---
if "retriever_video" in st.session_state and "chat_model_video" in st.session_state:
    for tipo, conteudo in st.session_state.memoria_video:
        with st.chat_message(tipo):
            st.markdown(conteudo)
            
    input_usuario = st.chat_input("Pergunte sobre o conteúdo do vídeo...")
    if input_usuario:
        st.session_state.memoria_video.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        chat_history_objects = [HumanMessage(content=c) if t == 'user' else AIMessage(content=c) for t, c in st.session_state.memoria_video]

        with st.chat_message("ai"):
            with st.spinner("Analisando a transcrição e gerando insights..."):
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa, gere uma pergunta de busca completa para a transcrição do vídeo."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model_video, st.session_state.retriever_video, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                contexto_formatado = "\n\n".join(doc.page_content for doc in docs_relevantes)
                
                prompt_final_str = f"""Você é um assistente especialista em análise de conteúdo de vídeos. Responda à pergunta do usuário com base na transcrição fornecida.
**Regras:**
1. Baseie sua resposta **exclusivamente** no 'Contexto da Transcrição do Vídeo' abaixo.
2. Se o contexto não tiver a resposta, diga claramente: "A informação não foi encontrada na transcrição do vídeo."
**Contexto da Transcrição do Vídeo:**
{contexto_formatado}
**Pergunta do Usuário:**
{input_usuario}
"""
                resposta_ai = st.session_state.chat_model_video.invoke(prompt_final_str).content
                st.markdown(resposta_ai)
        
        st.session_state.memoria_video.append(("ai", resposta_ai))
        st.rerun()
else:
    st.info("Aguardando configuração na barra lateral para iniciar a análise do vídeo.")