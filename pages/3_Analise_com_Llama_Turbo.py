# pages/3_Analise_com_Llama_Turbo.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

# Importações dos loaders que serão usados nesta página
from loaders import carrega_site, carrega_arquivo_upload

# Importações para a arquitetura RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain

# ===== INÍCIO DA CORREÇÃO: Importações Corretas para Hugging Face =====
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
# ===== FIM DA CORREÇÃO =====

# Carrega as variáveis de ambiente
load_dotenv()

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(page_title="Análise com Llama-Turbo", page_icon="🦙", layout="wide")

MODELO_LLAMA_TURBO = "meta-llama/Meta-Llama-3.1-405B-Instruct"
TIPOS_FONTES = ['Site', 'Pdf', 'Csv', 'Txt']

if "memoria_llama_turbo" not in st.session_state:
    st.session_state.memoria_llama_turbo = []

def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_chain_llama_turbo(tipo_fonte, fonte_dados):
    documento_bruto = ""
    if tipo_fonte == 'Site': documento_bruto = carrega_site(fonte_dados)
    elif tipo_fonte in ['Pdf', 'Csv', 'Txt']: documento_bruto = carrega_arquivo_upload(fonte_dados)
    
    if not documento_bruto:
        st.warning("O documento não pôde ser carregado."); return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.create_documents([documento_bruto])
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Chave da OpenAI é necessária para embeddings."); st.stop()
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not huggingface_api_key:
        st.error("Chave do Hugging Face (HUGGINGFACEHUB_API_TOKEN) não encontrada."); st.stop()
        
    # ===== INÍCIO DA CORREÇÃO: Inicialização em Duas Etapas =====
    # 1. Cria o "motor" LLM que se conecta ao endpoint
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=MODELO_LLAMA_TURBO,
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.5,
        max_new_tokens=2048,
    )

    # 2. Envolve o motor com a camada de chat
    chat_model = ChatHuggingFace(llm=llm_endpoint)
    # ===== FIM DA CORREÇÃO =====
    
    st.session_state.retriever_llama_turbo = retriever
    st.session_state.chat_model_llama_turbo = chat_model
    st.success("Analisador Llama-Turbo inicializado com sucesso!")


def pagina_chat_llama_turbo():
    st.title("🦙 Análise com Lhama-Turbo")
    st.caption(f"Página dedicada para análise de documentos com o modelo Llama 3.1 405B.")
    st.divider()
    
    if "retriever_llama_turbo" not in st.session_state or "chat_model_llama_turbo" not in st.session_state:
        st.info("Por favor, selecione uma fonte de dados e inicialize o analisador na barra lateral."); st.stop()

    for tipo, conteudo in st.session_state.memoria_llama_turbo:
        with st.chat_message(tipo):
            st.markdown(conteudo)
            
    input_usuario = st.chat_input("Faça sua pergunta ao Llama-Turbo...")
    if input_usuario:
        st.session_state.memoria_llama_turbo.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        chat_history_objects = []
        for tipo, conteudo in st.session_state.memoria_llama_turbo:
            if tipo == 'user': chat_history_objects.append(HumanMessage(content=conteudo))
            else: chat_history_objects.append(AIMessage(content=conteudo))

        with st.chat_message("ai"):
            with st.spinner("O Llama-Turbo está buscando e pensando..."):
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa acima, gere uma pergunta de busca completa e independente."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model_llama_turbo, st.session_state.retriever_llama_turbo, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                contexto_formatado = formatar_documentos(docs_relevantes)
                prompt_final_str = f"""Você é um assistente especialista usando o modelo Llama 3.1 405B. Responda à pergunta do usuário de forma completa, profunda e didática, com base no contexto fornecido.

**Contexto do Documento Fornecido:**
{contexto_formatado}

**Pergunta do Usuário:**
{input_usuario}
"""
                resposta_ai = st.session_state.chat_model_llama_turbo.invoke(prompt_final_str).content
                st.markdown(resposta_ai)
        
        st.session_state.memoria_llama_turbo.append(("ai", resposta_ai))
        st.rerun()

def construir_sidebar_llama_turbo():
    with st.sidebar:
        st.header("Analisador Llama-Turbo")
        
        tipo_fonte = st.selectbox("Selecione o tipo de fonte", TIPOS_FONTES, key="fonte_llama")
        fonte_dados = None
        if tipo_fonte == 'Site': fonte_dados = st.text_input("Digite a URL do site", key="url_llama")
        else: fonte_dados = st.file_uploader(f"Faça o upload do arquivo .{tipo_fonte.lower()}", type=[tipo_fonte.lower()], key="uploader_llama")
        
        st.divider()
        if st.button("🚀 Inicializar Análise", use_container_width=True, type="primary"):
            if fonte_dados:
                with st.spinner("Processando documento..."):
                    st.session_state.memoria_llama_turbo = []
                    inicializar_chain_llama_turbo(tipo_fonte, fonte_dados)
                    st.rerun()
            else:
                st.warning("Por favor, forneça uma fonte de dados.")
        
        if st.session_state.get('memoria_llama_turbo'):
            st.divider(); st.subheader("Gerenciar Conversa")
            if st.button("🗑️ Apagar Histórico", use_container_width=True, key="apagar_llama"):
                st.session_state.memoria_llama_turbo = []; st.rerun()
            
            historico_texto = ""
            for tipo, conteudo in st.session_state.memoria_llama_turbo:
                historico_texto += f"{tipo.capitalize()}: {conteudo}\n\n---\n\n"
            
            st.download_button(
                label="📥 Baixar Histórico (.txt)", data=historico_texto.encode('utf-8'),
                file_name=f"historico_llama_turbo_{int(time.time())}.txt", mime="text/plain",
                use_container_width=True, disabled=not historico_texto
            )

# --- Execução da Página ---
construir_sidebar_llama_turbo()
pagina_chat_llama_turbo()