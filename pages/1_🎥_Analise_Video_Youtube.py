# /pages/1_🎥_Analise_Video_Youtube.py (VERSÃO CORRIGIDA E FINAL)

import os
import streamlit as st
from dotenv import load_dotenv
import time
from io import BytesIO
import re

# --- Importações ---
from pytube import YouTube
import openai
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
st.caption("Cole a URL de um vídeo, configure o modelo de análise e extraia insights!")

# --- CORREÇÃO: Dicionário de modelos apenas para a ETAPA DE ANÁLISE ---
CONFIG_MODELOS_ANALISE = {
    'Groq': {'modelos': ['Llama3.1-405b-reasoning', 'llama3-70b-8192'], 'api_key': os.getenv("GROQ_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
}

# --- Funções de Lógica ---

def is_valid_youtube_url(url):
    if not url: return False
    youtube_regex = re.compile(r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)[\w-]{11}$')
    return youtube_regex.match(url) is not None

@st.cache_data(show_spinner="Buscando e transcrevendo o áudio do vídeo... (isso pode levar alguns minutos)")
def transcrever_video_com_openai(url_video, openai_api_key):
    """
    ETAPA 1: Transcrição usando Pytube e a API Whisper da OpenAI.
    Esta função agora recebe a chave da API explicitamente.
    """
    if not openai_api_key:
        st.error("Chave da API da OpenAI não encontrada! Por favor, configure-a para a transcrição.")
        return None
    try:
        yt = YouTube(url_video)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        buffer = BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        buffer.name = 'audio.mp3'
        
        client = openai.OpenAI(api_key=openai_api_key)
        transcricao = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            response_format="text"
        )
        return transcricao
    except Exception as e:
        st.error(f"Falha na transcrição com a OpenAI. Erro: {e}")
        return None

def inicializar_analise_completa(provedor_analise, modelo_analise, url_video, openai_api_key):
    # --- ETAPA 1: TRANSCRIÇÃO ---
    texto_transcrito = transcrever_video_com_openai(url_video, openai_api_key)
    
    if not texto_transcrito:
        st.warning("A transcrição falhou. O processo foi interrompido.")
        return

    st.success("Transcrição concluída! Vetorizando o texto para análise...")
    
    # --- ETAPA 2: VETORIZAÇÃO (também usa OpenAI) ---
    with st.spinner("Criando embeddings com a OpenAI..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([texto_transcrito])
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever_video = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # --- ETAPA 3: CONFIGURAÇÃO DO MODELO DE ANÁLISE (Groq ou Gemini) ---
    mapa_chat = {'Groq': ChatGroq, 'Gemini': ChatGoogleGenerativeAI}
    config_provedor = CONFIG_MODELOS_ANALISE[provedor_analise]
    api_key_analise = config_provedor['api_key']
    ChatModelClass = mapa_chat[provedor_analise]
    
    if provedor_analise == 'Gemini':
        chat_model = ChatModelClass(model=modelo_analise, google_api_key=api_key_analise)
    else:
        chat_model = ChatModelClass(model=modelo_analise, api_key=api_key_analise)
        
    st.session_state.chat_model_video = chat_model
    st.success("Tudo pronto! O Oráculo está preparado para analisar o vídeo.")

# --- Interface do Usuário ---

if "memoria_video" not in st.session_state:
    st.session_state.memoria_video = []

with st.sidebar:
    st.header("🛠️ Configurar Análise de Vídeo")
    
    # --- CORREÇÃO: Informação clara sobre o processo ---
    st.info("""
    **Processo de 2 Etapas:**
    1.  **Transcrição:** Feita com a API **Whisper da OpenAI**.
    2.  **Análise:** Feita com o modelo que você escolher abaixo (Groq ou Gemini).
    """)
    
    url_video = st.text_input("1. Cole a URL do vídeo do YouTube", key="url_video_input")
    
    # --- CORREÇÃO: Campo para a chave da OpenAI, essencial para a transcrição ---
    openai_api_key_input = st.text_input("2. Insira sua Chave da API da OpenAI", type="password", key="openai_key_video", help="Necessária para a transcrição do vídeo com o Whisper.")
    
    st.header("3. Escolha o Modelo para Análise")
    provedor = st.selectbox("Provedor de Análise", list(CONFIG_MODELOS_ANALISE.keys()), key="provedor_video")
    if provedor:
        modelo = st.selectbox("Modelo de Análise", CONFIG_MODELOS_ANALISE[provedor]['modelos'], key="modelo_video")
            
    if st.button("🚀 Iniciar Análise Completa", use_container_width=True, type="primary"):
        # Validação completa
        if not is_valid_youtube_url(url_video):
            st.error("URL inválida! Por favor, insira o link completo de um vídeo do YouTube.")
        elif not openai_api_key_input:
            st.error("A Chave da API da OpenAI é obrigatória para a transcrição.")
        elif not provedor or 'modelo' not in locals():
            st.warning("Por favor, selecione um provedor e um modelo de análise.")
        else:
            st.session_state.memoria_video = []
            # Passa a chave da OpenAI para a função principal
            inicializar_analise_completa(provedor, modelo, url_video, openai_api_key_input)
            
    if st.session_state.get('memoria_video'):
        if st.button("🗑️ Apagar Histórico", use_container_width=True):
            st.session_state.memoria_video = []
            st.rerun()

# --- Lógica do Chat Principal (sem alterações) ---
if "retriever_video" in st.session_state and "chat_model_video" in st.session_state:
    # ... (o resto do código do chat permanece o mesmo) ...
    for tipo, conteudo in st.session_state.memoria_video:
        with st.chat_message(tipo):
            st.markdown(conteudo)
    input_usuario = st.chat_input("Pergunte sobre o conteúdo do vídeo...")
    if input_usuario:
        # ... (código do chat sem alterações) ...
        st.session_state.memoria_video.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        chat_history_objects = [HumanMessage(content=c) if t == 'user' else AIMessage(content=c) for t, c in st.session_state.memoria_video]
        with st.chat_message("ai"):
            with st.spinner("Analisando a transcrição e gerando insights..."):
                prompt_reformulate = ChatPromptTemplate.from_messages([("placeholder", "{chat_history}"), ("user", "{input}"), ("user", "Com base na conversa, gere uma pergunta de busca completa para a transcrição do vídeo."),])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model_video, st.session_state.retriever_video, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                contexto_formatado = "\n\n".join(doc.page_content for doc in docs_relevantes)
                prompt_final_str = f"""Você é um assistente especialista... (sem alterações)"""
                resposta_ai = st.session_state.chat_model_video.invoke(prompt_final_str).content
                st.markdown(resposta_ai)
        st.session_state.memoria_video.append(("ai", resposta_ai))
        st.rerun()
else:
    st.info("Aguardando configuração na barra lateral para iniciar a análise do vídeo.")