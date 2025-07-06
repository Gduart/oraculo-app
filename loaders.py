# loaders.py

import os
import tempfile
import streamlit as st

# ===== INÍCIO DA CORREÇÃO: Importação Atualizada =====
# O GenericLoader foi movido para langchain_community.document_loaders.generic
from langchain_community.document_loaders.generic import GenericLoader 
# ===== FIM DA CORREÇÃO =====

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

def carrega_youtube(video_url: str) -> str:
    st.info(f"🔮 Iniciando download do áudio. Isso pode demorar...")
    st.warning("Este processo consome créditos da API OpenAI.")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Chave da API da OpenAI (OPENAI_API_KEY) não encontrada no seu arquivo .env para usar o Whisper.")
        return ""

    with tempfile.TemporaryDirectory() as save_dir:
        try:
            # O loader agora só baixa o áudio
            loader_audio = YoutubeAudioLoader([video_url], save_dir)
            # O parser que irá transcrever
            parser = OpenAIWhisperParser(api_key=openai_api_key, language="pt")

            # Usamos o GenericLoader para combinar o carregamento e a análise
            loader_generico = GenericLoader(loader_audio, parser)
            
            st.info("Download concluído. Enviando para transcrição com Whisper...")
            docs = loader_generico.load()
            
            if not docs:
                st.error("A transcrição com Whisper não retornou nenhum documento."); return ""
            
            st.success("✅ Vídeo transcrito com sucesso!")
            return _extrair_conteudo_documentos(docs)
        except Exception as e:
            st.error(f"❌ Ocorreu um erro crítico durante o download ou transcrição. Erro: {e}"); return ""

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