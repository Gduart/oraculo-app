import os
import streamlit as st
import subprocess
import sys
from dotenv import load_dotenv
import pyperclip
import time  # ### ADIÇÃO 1 de 4: Importado para usar no nome do arquivo de histórico

# --- LangChain Imports ---
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Provedores de Modelos e Componentes RAG ---
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carrega variáveis de ambiente
load_dotenv()

# --- Configurações da Página ---
st.set_page_config(page_title="Multi Análise IA", page_icon="🧩", layout="wide")
CONFIG_MODELOS = {
    'Groq': {'modelos': ['llama3-8b-8192', 'llama3-70b-8192'], 'api_key': os.getenv("GROQ_API_KEY")},
    'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o'], 'api_key': os.getenv("OPENAI_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
    'HuggingFace': {'modelos': ["Llama 3 8B", "Llama 3 70B"], 'api_key': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
}
HUGGINGFACE_REPO_IDS = { "Llama 3 8B": "meta-llama/Meta-Llama-3-8B-Instruct", "Llama 3 70B": "meta-llama/Meta-Llama-3-70B-Instruct" }
TIPOS_FONTES = ['Site', 'PDF', 'CSV', 'TXT'] # Adicionei Youtube aqui se você tiver a função carrega_youtube em loaders.py
if "memoria_analise" not in st.session_state: st.session_state.memoria_analise = []

# --- Funções Core com Arquitetura Final (LÓGICA FUNCIONAL MANTIDA) ---

def carrega_site(url: str) -> str:
    st.write(f"🤖 Acionando robô para analisar: {url}")
    pyperclip.copy("")
    python_executable = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scraper_script_path = os.path.join(project_root, 'scraper.py')
    if not os.path.exists(scraper_script_path):
        st.error(f"Erro Crítico: O script do robô não foi encontrado em '{scraper_script_path}'"); st.stop()
    try:
        subprocess.run([python_executable, scraper_script_path, url], capture_output=True, text=True, check=True, timeout=120)
        content = pyperclip.paste()
        if not content or not content.strip():
            st.error("O robô executou, mas não copiou nenhum conteúdo."); st.stop()
        if content.startswith("SCRAPER_ERROR:"):
            st.error(f"O robô de extração falhou: {content.replace('SCRAPER_ERROR:', '').strip()}"); st.stop()
        st.write("✅ Robô concluiu e resultado foi lido do clipboard com sucesso!")
        return content
    except subprocess.TimeoutExpired:
        st.error("A extração do site (robô) demorou demais (timeout de 120s)."); st.stop()
    except Exception as e:
        st.error(f"Um erro inesperado ocorreu ao chamar o robô: {str(e)}"); st.stop()

def carrega_arquivo_upload(arquivo_upload) -> str:
    nome_arquivo = arquivo_upload.name
    caminho_temp = os.path.join(".", f"temp_{nome_arquivo}")
    with open(caminho_temp, "wb") as f: f.write(arquivo_upload.getbuffer())
    try:
        if nome_arquivo.endswith('.pdf'): loader = PyPDFLoader(caminho_temp)
        elif nome_arquivo.endswith('.csv'): loader = CSVLoader(caminho_temp)
        elif nome_arquivo.endswith('.txt'): loader = TextLoader(caminho_temp, encoding='utf-8')
        else: st.error("Tipo de arquivo não suportado"); st.stop()
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)
    finally:
        if os.path.exists(caminho_temp): os.remove(caminho_temp)

def inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados):
    with st.spinner(f"Processando {tipo_fonte}..."):
        if tipo_fonte == 'Site': documento_bruto = carrega_site(fonte_dados)
        else: documento_bruto = carrega_arquivo_upload(fonte_dados)
        if not documento_bruto or not documento_bruto.strip():
            st.error("Não foi possível extrair conteúdo da fonte."); st.stop()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([documento_bruto])
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever_analise = vectorstore.as_retriever()
    with st.spinner(f"Inicializando modelo {modelo}..."):
        config = CONFIG_MODELOS[provedor]
        api_key = config['api_key']
        ChatClass = {'Groq': ChatGroq, 'OpenAI': ChatOpenAI, 'Gemini': ChatGoogleGenerativeAI, 'Deepseek': ChatDeepSeek}
        if provedor == 'HuggingFace':
            repo_id = HUGGINGFACE_REPO_IDS[modelo]
            llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=api_key)
            st.session_state.chat_model_analise = ChatHuggingFace(llm=llm)
        else:
            st.session_state.chat_model_analise = ChatClass[provedor](model=modelo, api_key=api_key)
    st.success("Analisador pronto! Pode fazer suas perguntas.")

# --- INTERFACE GRÁFICA (UI) ---
st.title("🔎 Análise Profunda com IA")
st.caption("Faça upload de um documento ou insira um site para extrair insights.")
with st.sidebar:
    st.header("🛠️ Configuração")
    tipo_fonte = st.selectbox("1. Escolha a Fonte", TIPOS_FONTES)
    fonte_dados = None
    if tipo_fonte == 'Site': fonte_dados = st.text_input("URL do Site", "https://openai.com/api/pricing/")
    else: fonte_dados = st.file_uploader(f"Upload do seu .{tipo_fonte.lower()}", type=[tipo_fonte.lower()])
    provedores_disponiveis = list(CONFIG_MODELOS.keys())
    provedor = st.selectbox("2. Escolha o Provedor de IA", provedores_disponiveis)
    modelo = st.selectbox("3. Escolha o Modelo", CONFIG_MODELOS[provedor]['modelos'])
    
    if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
        if not fonte_dados: st.warning("Por favor, forneça uma fonte de dados.")
        else:
            inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados)
            st.session_state.memoria_analise = []
            st.rerun()

    # ### ADIÇÃO 2 de 4: BOTÕES DE GERENCIAMENTO DO HISTÓRICO ###
    if "retriever_analise" in st.session_state:
        st.divider()
        st.subheader("Gerenciar Conversa")
        if st.button("🗑️ Apagar Histórico", use_container_width=True):
            st.session_state.memoria_analise = []
            st.rerun()

        historico_texto = ""
        for tipo, conteudo in st.session_state.memoria_analise:
            historico_texto += f"{tipo.capitalize()}: {conteudo}\n\n---\n\n"
        
        st.download_button(
            label="📥 Baixar Histórico (.txt)",
            data=historico_texto.encode('utf-8'),
            file_name=f"historico_analise_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not historico_texto
        )

# ### ADIÇÃO 3 de 4: CONTÊINER COM BARRA DE ROLAGEM PARA O CHAT ###
chat_container = st.container(height=600)
with chat_container:
    if "retriever_analise" in st.session_state:
        for tipo, conteudo in st.session_state.memoria_analise:
            with st.chat_message(tipo): st.markdown(conteudo)
    else:
        st.info("Configure a fonte de dados e o modelo na barra lateral para começar a análise.")

if "retriever_analise" in st.session_state:
    if prompt := st.chat_input("Pergunte sobre o documento..."):
        st.session_state.memoria_analise.append(("user", prompt))
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                chat_history = [HumanMessage(content=c) if t=="user" else AIMessage(content=c) for t,c in st.session_state.memoria_analise]
                retriever_chain = create_history_aware_retriever(st.session_state.chat_model_analise, st.session_state.retriever_analise, ChatPromptTemplate.from_messages([("placeholder", "{chat_history}"),("user", "{input}"), ("user", "Reformule a pergunta para busca")]))
                docs = retriever_chain.invoke({"chat_history": chat_history, "input": prompt})
                
                # ### ADIÇÃO 4 de 4: AJUSTE NO PROMPT PARA RESPOSTA EM PORTUGUÊS ###
                system_prompt_template = """Responda à pergunta do usuário baseado no seguinte contexto. Responda sempre em Português do Brasil.
Contexto:
{context}"""
                
                rag_chain = create_stuff_documents_chain(st.session_state.chat_model_analise, ChatPromptTemplate.from_messages([("system", system_prompt_template), ("placeholder", "{chat_history}"), ("user", "{input}")]))
                
                resposta = rag_chain.invoke({"context": docs, "chat_history": chat_history, "input": prompt})
                st.markdown(resposta)
        st.session_state.memoria_analise.append(("assistant", resposta))
        st.rerun()