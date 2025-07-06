# /pages/1_🎥_Analise_Video_Youtube.py (CÓDIGO DE DIAGNÓSTICO FINAL)

import os
import streamlit as st
from io import BytesIO
import re
from pytube import YouTube, exceptions as PytubeExceptions
import openai

# --- Carregando as chaves de forma segura ---
# O Streamlit lê isso dos "Secrets" que você configurou
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Diagnóstico de Transcrição", page_icon="🔬", layout="wide")
st.title("🔬 Página de Diagnóstico de Transcrição")
st.write("Esta página serve apenas para testar a conexão e a transcrição de forma isolada.")

# --- Interface de Teste ---
st.header("Insira os dados para o teste")

url_video = st.text_input(
    "URL do Vídeo do YouTube (use um vídeo público conhecido)", 
    value="https://www.youtube.com/3" # Exemplo: Vídeo sobre o Llama 3.1
)

st.info(f"Chave da OpenAI carregada dos Secrets? {'Sim, encontrada.' if OPENAI_API_KEY and OPENAI_API_KEY.startswith('sk-') else 'NÃO, não encontrada ou inválida!'}")

if st.button("▶️ EXECUTAR DIAGNÓSTICO", use_container_width=True, type="primary"):

    st.header("Resultados do Diagnóstico em Tempo Real")
    
    # Validação da URL
    youtube_regex = re.compile(r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}.*')
    if not youtube_regex.match(url_video):
        st.error("FALHA NA VALIDAÇÃO: A URL não é um link de vídeo válido do YouTube.")
        st.stop()
    st.success("Passo 1/5: Validação da URL... OK")

    # Tentativa de conexão
    try:
        with st.spinner("Tentando conectar ao YouTube e obter o título do vídeo..."):
            yt = YouTube(url_video)
            st.success(f"Passo 2/5: Conexão com YouTube... OK. Título: '{yt.title}'")

        with st.spinner("Localizando e baixando o fluxo de áudio para a memória..."):
            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                st.error("FALHA: Não foi encontrado um fluxo de áudio para este vídeo.")
                st.stop()

            buffer = BytesIO()
            audio_stream.stream_to_buffer(buffer)
            buffer.seek(0)
            buffer.name = 'audio.mp3'
            st.success("Passo 3/5: Download do áudio para memória... OK")

        with st.spinner("Enviando o áudio para a API Whisper da OpenAI..."):
            if not OPENAI_API_KEY:
                st.error("FALHA: Chave da API da OpenAI não encontrada nos Secrets. Impossível continuar.")
                st.stop()
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer,
                response_format="text"
            )
            st.success("Passo 4/5: Transcrição via API OpenAI... OK")

        st.balloons()
        st.success("PASSO 5/5: PROCESSO CONCLUÍDO COM SUCESSO!")
        st.subheader("Texto Transcrito:")
        st.text_area("Resultado", transcription, height=300)

    except Exception as e:
        st.error("❌ O PROCESSO FALHOU. ABAIXO ESTÁ O ERRO TÉCNICO EXATO:")
        st.exception(e)