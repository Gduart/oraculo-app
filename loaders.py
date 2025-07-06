# loaders.py

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders.generic import GenericLoader 
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

def _extrair_conteudo_documentos(lista_documentos):
    if not lista_documentos: return ""
    return '\n\n'.join([doc.page_content for doc in lista_documentos])

def carrega_site(url: str) -> str:
    st.info(f"Analisando o conteúdo da URL: {url}...")
    try:
        loader = WebBaseLoader(web_path=url)
        return _extrair_conteudo_documentos(loader.load())
    except Exception as e:
        st.error(f"Não foi possível carregar o site. Erro: {e}"); return ""

# ===== INÍCIO DA CORREÇÃO DEFINITIVA =====
def carrega_youtube(video_url: str) -> str:
    st.info(f"🔮 Iniciando download do áudio...")
    st.warning("Este processo pode demorar e consome créditos da API OpenAI.")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Chave da API da OpenAI (OPENAI_API_KEY) não foi encontrada no .env.")
        return ""

    with tempfile.TemporaryDirectory() as save_dir:
        try:
            # Opções explícitas para o downloader para evitar pós-processamento pesado
            downloader_options = {
                "format": "m4a/bestaudio/best", # Pega o melhor áudio já em formato m4a
                "nopostoverwrites": True,      # Não sobrescreve se já existe
                "fixup": "never"               # NUNCA tenta "consertar" o arquivo
            }

            loader_audio = YoutubeAudioLoader(
                [video_url], 
                save_dir,
                downloader_options=downloader_options
            )
            
            parser = OpenAIWhisperParser(api_key=openai_api_key, language="pt")
            loader_generico = GenericLoader(loader_audio, parser)
            
            st.info("Download concluído. Enviando para transcrição...")
            docs = loader_generico.load()
            
            if not docs:
                st.error("A transcrição com Whisper não retornou nenhum documento."); return ""
            
            st.success("✅ Vídeo transcrito com sucesso!")
            return _extrair_conteudo_documentos(docs)
        except Exception as e:
            st.error(f"❌ Ocorreu um erro crítico durante o processo. Erro: {e}"); return ""
# ===== FIM DA CORREÇÃO DEFINITIVA =====

def carrega_arquivo_upload(arquivo_uploader) -> str:
    st.info(f"🔮 Analisando o arquivo: {arquivo_uploader.name}...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(arquivo_uploader.name)[1]) as temp_file:
            temp_file.write(arquivo_uploader.getvalue())
            caminho_arquivo_temp = temp_file.name
        extensao = os.path.splitext(caminho_arquivo_temp)[1].lower()
        if extensao == '.pdf': loader = PyPDFLoader(caminho_arquivo_temp)
        elif extensao == '.csv': loader = CSVLoader(caminho_arquivo_temp)
        elif extensao == '.txt': loader = TextLoader(caminho_arquivo_temp)
        else:
            st.error(f"Tipo de arquivo '{extensao}' não suportado.")
            if os.path.exists(caminho_arquivo_temp): os.remove(caminho_arquivo_temp)
            return ""
        documentos = loader.load()
        if os.path.exists(caminho_arquivo_temp): os.remove(caminho_arquivo_temp)
        return _extrair_conteudo_documentos(documentos)
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Erro: {e}"); return ""