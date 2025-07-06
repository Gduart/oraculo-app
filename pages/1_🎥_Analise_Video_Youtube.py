# /pages/1_🎥_Analise_Video_Youtube.py

import os
import streamlit as st
from dotenv import load_dotenv
import time
from io import BytesIO

# --- Nossas Novas Importações Robustas ---
from pytube import YouTube
import openai

# --- Importações do LangChain para Análise ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Carregar variáveis de ambiente
load_dotenv()

# --- Configurações Iniciais ---
st.set_page_config(page_title="Análise de Vídeo", page_icon="🎥", layout="wide")
st.title("🎥 Oráculo - Análise de Vídeo do YouTube")
st.caption("Cole a URL do vídeo, configure o modelo de análise e extraia insights valiosos!")

# Modelos para a análise (conforme solicitado por você)
CONFIG_MODELOS = {
    'Groq': {'modelos': ['Llama3.1-405b-reasoning', 'llama3-70b-8192'], 'api_key': os.getenv("GROQ_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
}

# --- Funções de Lógica ---

@st.cache_data(show_spinner="Buscando e transcrevendo o áudio do vídeo... (isso pode levar alguns minutos)")
def transcrever_via_api(url_video):
    """
    Função robusta que usa Pytube para obter o áudio em memória e a API Whisper da OpenAI para transcrever.
    """
    try:
        yt = YouTube(url_video)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        buffer = BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        buffer.name = 'audio.mp3' # A API precisa de um nome de arquivo
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        transcricao = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            response_format="text"
        )
        return transcricao
    except Exception as e:
        st.error(f"Falha ao processar o vídeo: {e}")
        return None

def inicializar_analise_video(provedor, modelo, url_video):
    # 1. Transcrever usando a nova função robusta
    texto_transcrito = transcrever_via_api(url_video)
    
    if not texto_transcrito:
        st.warning("A transcrição falhou. Verifique a URL do vídeo ou tente novamente.")
        return

    # 2. Vetorizar a transcrição (padrão LangChain)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.create_documents([texto_transcrito])
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.retriever_video = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Configurar o LLM de análise
    mapa_chat = {'Groq': ChatGroq, 'Gemini': ChatGoogleGenerativeAI}
    api_key = CONFIG_MODELOS[provedor]['api_key']
    ChatModelClass = mapa_chat[provedor]
    
    if provedor == 'Gemini':
        chat_model = ChatModelClass(model=modelo, google_api_key=api_key)
    else:
        chat_model = ChatModelClass(model=modelo, api_key=api_key)
        
    st.session_state.chat_model_video = chat_model
    st.success("Análise de vídeo pronta! Pode fazer suas perguntas.")

# --- Interface do Usuário (Sidebar e Chat) ---

if "memoria_video" not in st.session_state:
    st.session_state.memoria_video = []

with st.sidebar:
    st.header("🛠️ Configurar Análise de Vídeo")
    url_video = st.text_input("Cole a URL do vídeo do YouTube", key="url_video_input")
    
    provedor = st.selectbox("Provedor de Análise", list(CONFIG_MODELOS.keys()), key="provedor_video")
    if provedor:
        modelo = st.selectbox("Modelo de Análise", CONFIG_MODELOS[provedor]['modelos'], key="modelo_video")
            
    if st.button("🚀 Iniciar Análise do Vídeo", use_container_width=True, type="primary"):
        if url_video and provedor and 'modelo' in locals():
            st.session_state.memoria_video = []
            inicializar_analise_video(provedor, modelo, url_video)
        else:
            st.warning("Por favor, preencha a URL e selecione um modelo.")
            
    if st.session_state.get('memoria_video'):
        if st.button("🗑️ Apagar Histórico", use_container_width=True):
            st.session_state.memoria_video = []
            st.rerun()

# --- Lógica do Chat Principal ---

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
                
                # Mesma lógica robusta de recuperação de contexto que você já usa
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa, gere uma pergunta de busca completa para a transcrição do vídeo."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model_video, st.session_state.retriever_video, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                contexto_formatado = "\n\n".join(doc.page_content for doc in docs_relevantes)
                
                prompt_final_str = f"""Você é um assistente especialista em análise de conteúdo de vídeos. Responda à pergunta do usuário com base na transcrição fornecida.

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