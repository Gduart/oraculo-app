# Oraculo.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import time

from loaders import carrega_site, carrega_youtube, carrega_arquivo_upload

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
st.set_page_config(page_title="Oráculo", page_icon="🔮", layout="wide")

CONFIG_MODELOS = {
    'Groq': {'modelos': ['llama3-70b-8192', 'llama3-8b-8192', 'gemma2-9b-it','Llama 3.1 405B BASE'], 'api_key': os.getenv("GROQ_API_KEY")},
    'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini'], 'api_key': os.getenv("OPENAI_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest','gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
    'Deepseek': {'modelos': ['deepseek-chat', 'deepseek-coder'], 'api_key': os.getenv("DEEPSEEK_API_KEY")}
}
TIPOS_FONTES = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

if "memoria" not in st.session_state:
    st.session_state.memoria = []

def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_chain_e_memoria(provedor, modelo, tipo_fonte, fonte_dados):
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
    from langchain_deepseek.chat_models import ChatDeepSeek
    mapa_chat = {'Groq': ChatGroq, 'OpenAI': ChatOpenAI, 'Gemini': ChatGoogleGenerativeAI, 'Deepseek': ChatDeepSeek}
    
    documento_bruto = ""
    if tipo_fonte == 'Site': documento_bruto = carrega_site(fonte_dados)
    elif tipo_fonte == 'Youtube': documento_bruto = carrega_youtube(fonte_dados)
    elif tipo_fonte in ['Pdf', 'Csv', 'Txt']: documento_bruto = carrega_arquivo_upload(fonte_dados)
    
    if not documento_bruto:
        st.warning("O documento não pôde ser carregado."); return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.create_documents([documento_bruto])
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Chave da OpenAI necessária para embeddings."); st.stop()
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    config_provedor = CONFIG_MODELOS[provedor]
    api_key = config_provedor['api_key']
    if not api_key: st.error(f"Chave de API para {provedor} não encontrada."); st.stop()
    ChatModelClass = mapa_chat[provedor]
    
    if provedor == 'Gemini':
        chat_model = ChatModelClass(model=modelo, google_api_key=api_key)
    else:
        chat_model = ChatModelClass(model=modelo, api_key=api_key)
    
    st.session_state.retriever = retriever
    st.session_state.chat_model = chat_model
    st.success("Oráculo inicializado com sucesso!")

def pagina_principal_chat():
    st.title("🔮 Oráculo - Insights Poderosos")
    st.caption("Escolha uma fonte de dados, configure o modelo na barra lateral e comece a interagir.")
    st.divider()
    
    if "retriever" not in st.session_state or "chat_model" not in st.session_state:
        st.info("Por favor, configure e inicialize o Oráculo na barra lateral para começar."); st.stop()

    for tipo, conteudo in st.session_state.memoria:
        with st.chat_message(tipo):
            st.markdown(conteudo)
            
    input_usuario = st.chat_input("Faça sua pergunta ao Oráculo...")
    if input_usuario:
        st.session_state.memoria.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        chat_history_objects = []
        for tipo, conteudo in st.session_state.memoria:
            if tipo == 'user': chat_history_objects.append(HumanMessage(content=conteudo))
            else: chat_history_objects.append(AIMessage(content=conteudo))

        with st.chat_message("ai"):
            with st.spinner("O Oráculo está buscando e pensando..."):
                # ===== INÍCIO DA CORREÇÃO: FLUXO MANUAL E ROBUSTO =====
                
                # 1. Reformular a pergunta com base no histórico
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa acima, gere uma pergunta de busca completa e independente para encontrar informações relevantes no documento."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model, st.session_state.retriever, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                # 2. Construir o prompt final manualmente
                contexto_formatado = formatar_documentos(docs_relevantes)
                prompt_final_str = f"""Você é um assistente especialista chamado Oráculo. Responda à pergunta do usuário de forma completa e didática.

Regras:
1.  Sua fonte de verdade é o 'Contexto do Documento' abaixo. Baseie sua resposta nele.
2.  Se o contexto não for suficiente, use seu conhecimento geral para complementar, mas avise que a informação extra veio de você.
3.  Quando o usuário falar de 'vídeo' ou 'site', ele se refere ao contexto.

**Contexto do Documento Fornecido:**
{contexto_formatado}

**Pergunta do Usuário:**
{input_usuario}
"""
                # 3. Invocar o modelo com a lista de mensagens limpa
                resposta_ai = st.session_state.chat_model.invoke(prompt_final_str).content
                st.markdown(resposta_ai)
                # ===== FIM DA CORREÇÃO =====
        
        st.session_state.memoria.append(("ai", resposta_ai))
        st.rerun()

def construir_sidebar():
    with st.sidebar:
        st.header("🛠️ Configurações do Oráculo")
        with st.expander("1. Fonte de Dados", expanded=True):
            tipo_fonte = st.selectbox("Selecione o tipo", TIPOS_FONTES)
            fonte_dados = None
            if tipo_fonte == 'Site': fonte_dados = st.text_input("Digite a URL do site")
            elif tipo_fonte == 'Youtube': fonte_dados = st.text_input("Digite a URL do vídeo")
            else: fonte_dados = st.file_uploader(f"Faça o upload do arquivo .{tipo_fonte.lower()}", type=[tipo_fonte.lower()])
        
        with st.expander("2. Escolha do Modelo", expanded=True):
            provedor = st.selectbox("Selecione o Provedor", list(CONFIG_MODELOS.keys()))
            if provedor: modelo = st.selectbox("Selecione o Modelo", CONFIG_MODELOS[provedor]['modelos'])
        
        st.divider()
        if st.button("🚀 Inicializar Oráculo", use_container_width=True, type="primary"):
            if fonte_dados and provedor and 'modelo' in locals():
                with st.spinner("Processando o documento..."):
                    st.session_state.memoria = []
                    inicializar_chain_e_memoria(provedor, modelo, tipo_fonte, fonte_dados)
                    st.rerun()
            else: st.warning("Por favor, forneça uma fonte de dados e selecione um modelo.")
        
        if st.session_state.get('memoria'):
            st.divider(); st.subheader("Gerenciar Conversa")
            if st.button("🗑️ Apagar Histórico", use_container_width=True):
                st.session_state.memoria = []; st.rerun()
            historico_texto = ""
            for tipo, conteudo in st.session_state.memoria:
                historico_texto += f"{tipo.capitalize()}: {conteudo}\n\n---\n\n"
            st.download_button(
                label="📥 Baixar Histórico (.txt)", data=historico_texto.encode('utf-8'),
                file_name=f"historico_oraculo_{int(time.time())}.txt", mime="text/plain",
                use_container_width=True, disabled=not historico_texto
            )

# Removemos o "if __name__ == '__main__':" para seguir o padrão de apps multipágina
construir_sidebar()
pagina_principal_chat()

#Teste1