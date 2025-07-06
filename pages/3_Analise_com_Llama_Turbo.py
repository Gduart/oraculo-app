# Analisador.py (Baseado no Oraculo.py original)

import os
import streamlit as st
from dotenv import load_dotenv
import time

# --- LangChain Imports ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Provedores de Modelos ---
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace

# --- Componentes RAG ---
# Assumindo que seu arquivo loaders.py tem estas funções
from loaders import carrega_site, carrega_arquivo_upload
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carrega variáveis de ambiente
load_dotenv()

# --- Configurações da Página e Globais ---
st.set_page_config(page_title="Analisador de Documentos", page_icon="🧩", layout="wide")

# Dicionário de configuração dos provedores e seus modelos
CONFIG_MODELOS = {
    'HuggingFace': {
        'modelos': ["Llama 3.1 8B (Gratuito)", "Llama 3 70B (Gratuito)", "Llama 3.1 405B (Pago)"], 
        'api_key': os.getenv("HUGGINGFACEHUB_API_TOKEN")
    },
    'Groq': {
        'modelos': ['llama3-8b-8192', 'llama3-70b-8192', 'gemma2-9b-it'], 
        'api_key': os.getenv("GROQ_API_KEY")
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o'], 
        'api_key': os.getenv("OPENAI_API_KEY")
    },
    'Gemini': {
        'modelos': ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'], 
        'api_key': os.getenv("GOOGLE_API_KEY")
    },
    'Deepseek': {
        'modelos': ['deepseek-chat', 'deepseek-coder'], 
        'api_key': os.getenv("DEEPSEEK_API_KEY")
    }
}

HUGGINGFACE_REPO_IDS = {
    "Llama 3.1 8B (Gratuito)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama 3 70B (Gratuito)": "meta-llama/Meta-Llama-3-70B-Instruct",
    "Llama 3.1 405B (Pago)": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}

# Tipos de fontes, sem Youtube
TIPOS_FONTES = ['Site', 'Pdf', 'Csv', 'Txt']

# Inicializa o estado da sessão
if "memoria" not in st.session_state:
    st.session_state.memoria = []

def formatar_documentos(docs):
    """Formata a lista de documentos em uma única string."""
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados):
    """Função central que carrega dados, cria a VectorStore e inicializa a cadeia RAG."""
    mapa_chat = {
        'Groq': ChatGroq, 'OpenAI': ChatOpenAI, 'Gemini': ChatGoogleGenerativeAI, 
        'Deepseek': ChatDeepSeek, 'HuggingFace': ChatHuggingFace
    }
    
    with st.spinner("Carregando e processando o documento..."):
        documento_bruto = ""
        if tipo_fonte == 'Site': documento_bruto = carrega_site(fonte_dados)
        elif tipo_fonte in ['Pdf', 'Csv', 'Txt']: documento_bruto = carrega_arquivo_upload(fonte_dados)
        
        if not documento_bruto:
            st.warning("O documento não pôde ser carregado."); return
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([documento_bruto])
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key: st.error("Chave da OpenAI é necessária para embeddings."); st.stop()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    with st.spinner(f"Inicializando modelo '{modelo}'..."):
        config_provedor = CONFIG_MODELOS[provedor]
        api_key = config_provedor['api_key']
        if not api_key: st.error(f"Chave de API para {provedor} não encontrada."); st.stop()
        
        ChatModelClass = mapa_chat[provedor]
        chat_model = None

        if provedor == 'HuggingFace':
            repo_id = HUGGINGFACE_REPO_IDS[modelo]
            llm_endpoint = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=api_key, temperature=0.3)
            chat_model = ChatHuggingFace(llm=llm_endpoint)
        elif provedor == 'Gemini':
            chat_model = ChatModelClass(model=modelo, google_api_key=api_key)
        else:
            chat_model = ChatModelClass(model=modelo, api_key=api_key)
        
        st.session_state.chat_model = chat_model
    
    st.success("Analisador inicializado com sucesso!")

def pagina_principal():
    st.title("🧩 Analisador de Documentos Multi-IA")
    st.caption("Converse com seus documentos (PDF, CSV, TXT ou Sites)")
    st.divider()
    
    if "retriever" not in st.session_state or "chat_model" not in st.session_state:
        st.info("Por favor, configure e inicialize o analisador na barra lateral para começar."); st.stop()

    for tipo, conteudo in st.session_state.memoria:
        with st.chat_message(tipo):
            st.markdown(conteudo)
            
    input_usuario = st.chat_input("Faça sua pergunta sobre o documento...")
    if input_usuario:
        st.session_state.memoria.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        chat_history_objects = [HumanMessage(content=c) if t == 'user' else AIMessage(content=c) for t, c in st.session_state.memoria]

        with st.chat_message("ai"):
            with st.spinner("A IA está buscando e pensando..."):
                # LÓGICA RAG ORIGINAL E ESTÁVEL
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa, gere uma pergunta de busca para encontrar informações relevantes."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model, st.session_state.retriever, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                contexto_formatado = formatar_documentos(docs_relevantes)
                
                prompt_final = ChatPromptTemplate.from_messages([
                    ("system", """Você é um assistente especialista em análise de documentos. Responda à pergunta do usuário de forma completa e didática.

Regras:
1. Sua fonte de verdade é o 'Contexto do Documento' abaixo. Baseie sua resposta nele.
2. Se o contexto não for suficiente, use seu conhecimento geral para complementar, mas avise que a informação extra veio de você.

**Contexto do Documento Fornecido:**
{context}"""),
                    ("placeholder", "{chat_history}"),
                    ("user", "{input}")
                ])
                
                document_chain = create_stuff_documents_chain(st.session_state.chat_model, prompt_final)
                
                resposta_ai = document_chain.invoke({
                    "chat_history": chat_history_objects,
                    "input": input_usuario,
                    "context": docs_relevantes
                })

                st.markdown(resposta_ai)
        
        st.session_state.memoria.append(("ai", resposta_ai))
        st.rerun()

def construir_sidebar():
    with st.sidebar:
        st.header("🛠️ Configurações")
        with st.expander("1. Fonte de Dados", expanded=True):
            tipo_fonte = st.selectbox("Selecione o tipo de fonte", TIPOS_FONTES)
            fonte_dados = None
            if tipo_fonte == 'Site':
                fonte_dados = st.text_input("Digite a URL do site")
            else:
                fonte_dados = st.file_uploader(f"Faça o upload do arquivo .{tipo_fonte.lower()}", type=[tipo_fonte.lower()])
        
        with st.expander("2. Escolha do Modelo de IA", expanded=True):
            provedor = st.selectbox("Selecione o Provedor", list(CONFIG_MODELOS.keys()))
            modelo = st.selectbox("Selecione o Modelo", CONFIG_MODELOS[provedor]['modelos']) if provedor else ""
        
        st.divider()
        if st.button("🚀 Iniciar Análise", use_container_width=True, type="primary"):
            if fonte_dados and provedor and modelo:
                st.session_state.memoria = []
                inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados)
                st.rerun()
            else:
                st.warning("Por favor, forneça uma fonte de dados e selecione um modelo.")
        
        if st.session_state.get('memoria'):
            st.divider()
            st.subheader("Gerenciar Conversa")
            if st.button("🗑️ Limpar Histórico", use_container_width=True):
                st.session_state.memoria = []; st.rerun()
            
            historico_texto = "\n\n---\n\n".join([f"{tipo.capitalize()}: {conteudo}" for tipo, conteudo in st.session_state.memoria])
            st.download_button(
                label="📥 Baixar Histórico (.txt)", 
                data=historico_texto.encode('utf-8'),
                file_name=f"historico_analisador_{int(time.time())}.txt", 
                mime="text/plain",
                use_container_width=True, 
                disabled=not historico_texto
            )

# --- Execução Principal ---
construir_sidebar()
pagina_principal()