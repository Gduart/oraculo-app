# Oraculo.py (Versão Final com sua lógica preservada + correções de bugs)

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

# Funções de carregamento importadas do seu arquivo loaders.py
from loaders import carrega_site, carrega_youtube, carrega_arquivo_upload

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_deepseek.chat_models import ChatDeepSeek

load_dotenv()
st.set_page_config(page_title="Oráculo", page_icon="🔮", layout="wide")

# --- CONFIGURAÇÕES ---
CONFIG_MODELOS = {
    'Groq': {'modelos': ['llama3-70b-8192', 'llama3-8b-8192', 'gemma2-9b-it','Llama 3.1 405B BASE'], 'api_key': os.getenv("GROQ_API_KEY")},
    'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini'], 'api_key': os.getenv("OPENAI_API_KEY")},
    'Gemini': {'modelos': ['gemini-1.5-pro-latest','gemini-1.5-flash-latest'], 'api_key': os.getenv("GOOGLE_API_KEY")},
    'Deepseek': {'modelos': ['deepseek-chat', 'deepseek-coder'], 'api_key': os.getenv("DEEPSEEK_API_KEY")}
}
TIPOS_FONTES = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

# --- Inicialização isolada do estado de sessão para esta página ---
if "oraculo_memoria" not in st.session_state:
    st.session_state.oraculo_memoria = []
if "oraculo_retriever" not in st.session_state:
    st.session_state.oraculo_retriever = None
if "oraculo_chat_model" not in st.session_state:
    st.session_state.oraculo_chat_model = None

def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados):
    mapa_chat = {'Groq': ChatGroq, 'OpenAI': ChatOpenAI, 'Gemini': ChatGoogleGenerativeAI, 'Deepseek': ChatDeepSeek}
    
    with st.spinner("Processando o documento..."):
        documento_bruto = ""
        # Lógica original de carregamento de fontes mantida
        if tipo_fonte == 'Site': documento_bruto = carrega_site(fonte_dados)
        elif tipo_fonte == 'Youtube': documento_bruto = carrega_youtube(fonte_dados)
        elif tipo_fonte in ['Pdf', 'Csv', 'Txt']: documento_bruto = carrega_arquivo_upload(fonte_dados)
    
    if not documento_bruto:
        st.warning("O documento não pôde ser carregado."); return
    
    with st.spinner("Criando base de conhecimento..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.create_documents([documento_bruto])
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    with st.spinner(f"Inicializando modelo {modelo}..."):
        config_provedor = CONFIG_MODELOS[provedor]
        api_key = config_provedor['api_key']
        if not api_key: st.error(f"Chave de API para {provedor} não encontrada."); st.stop()
        
        ChatModelClass = mapa_chat[provedor]
        if provedor == 'Gemini':
            chat_model = ChatModelClass(model=modelo, google_api_key=api_key)
        else:
            chat_model = ChatModelClass(model=modelo, api_key=api_key)
    
    # Salvando no estado de sessão com nomes únicos para evitar conflito entre páginas
    st.session_state.oraculo_retriever = retriever
    st.session_state.oraculo_chat_model = chat_model
    st.session_state.oraculo_memoria = []
    st.success("Oráculo inicializado com sucesso!")

def pagina_principal_chat():
    st.title("🔮 Oráculo - Insights Poderosos")
    st.caption("Escolha uma fonte de dados, configure o modelo na barra lateral e comece a interagir.")
    st.divider()
    
    # Verifica o estado de sessão correto para esta página
    if not st.session_state.oraculo_retriever or not st.session_state.oraculo_chat_model:
        st.info("Por favor, configure e inicialize o Oráculo na barra lateral para começar."); return

    for tipo, conteudo in st.session_state.oraculo_memoria:
        with st.chat_message(tipo):
            st.markdown(conteudo)
            
    input_usuario = st.chat_input("Faça sua pergunta ao Oráculo...")
    if input_usuario:
        st.session_state.oraculo_memoria.append(("user", input_usuario))
        with st.chat_message("user"):
            st.markdown(input_usuario)
        
        chat_history_objects = [HumanMessage(content=c) if t == 'user' else AIMessage(content=c) for t, c in st.session_state.oraculo_memoria]

        with st.chat_message("ai"):
            with st.spinner("O Oráculo está buscando e pensando..."):
                # Lógica RAG original mantida
                prompt_reformulate = ChatPromptTemplate.from_messages([
                    ("placeholder", "{chat_history}"), ("user", "{input}"),
                    ("user", "Com base na conversa acima, gere uma pergunta de busca completa para o documento."),
                ])
                retriever_chain = create_history_aware_retriever(st.session_state.oraculo_chat_model, st.session_state.oraculo_retriever, prompt_reformulate)
                docs_relevantes = retriever_chain.invoke({"chat_history": chat_history_objects, "input": input_usuario})
                
                # A única alteração é nas instruções do prompt para a IA não se confundir
                system_prompt_template = """Você é um Especilista em Analises de variados tipos de Documentos chamado Oráculo. Sua única função é ANALISAR e trazer Insights poderosos, organizados e que vai gerar valor ao usuario, retirado do conteúdo de texto fornecido no 'Contexto do Documento'.

REGRAS CRÍTICAS E INEGOCIÁVEIS:
1. FOCO NO CONTEÚDO DE TEXTO: Responda à pergunta do usuário focando exclusivamente no texto em prosa, dados e informações do 'Contexto do Documento'.
2. IGNORE CÓDIGOS DE PROGRAMAÇÃO: Se o contexto contiver exemplos de código (Python, etc.), IGNORE-OS COMPLETAMENTE. Sua tarefa é analisar o significado do texto, não explicar o código. Não escreva código na sua resposta.
3. SEJA UM ESPECIALISTA, NÃO UM PROGRAMADOR: Aja sempre como um especialista em analises que interpreta informações, de acordo com o documento enviado ou solicitado e desenvolver junto com sua Especialidade, o melhor insight. Nunca aja como um programador.
4. FONTE DA VERDADE ÚNICA: O 'Contexto do Documento' é sua única fonte de informação. Se a resposta não estiver lá, afirme claramente que "A informação não foi encontrada no documento fornecido".
5. IDIOMA: Responda sempre em Português do Brasil.
6. CASO SEJA PRECISO: Pode completar com sua Especialidade Multiplas em diversos assuntos para completar a informação do documento, caso a informação da fonte seja irrelevante.

**Contexto do Documento Fornecido:**
{context}
"""
                qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt_template), ("human", "{input}")])
                
                # Lógica RAG original mantida
                Youtube_chain = create_stuff_documents_chain(st.session_state.oraculo_chat_model, qa_prompt)
                
                resposta_ai = Youtube_chain.invoke({
                    "input": input_usuario,
                    "context": docs_relevantes,
                    "chat_history": chat_history_objects
                })
                
                st.markdown(resposta_ai)
        
        st.session_state.oraculo_memoria.append(("ai", resposta_ai))
        st.rerun()

def construir_sidebar():
    with st.sidebar:
        st.header("🛠️ Configurações do Oráculo")
        with st.expander("1. Fonte de Dados", expanded=True):
            tipo_fonte = st.selectbox("Selecione o tipo", TIPOS_FONTES, key="oraculo_tipo_fonte")
            fonte_dados = None
            if tipo_fonte == 'Site': fonte_dados = st.text_input("Digite a URL do site", key="oraculo_site")
            elif tipo_fonte == 'Youtube': fonte_dados = st.text_input("Digite a URL do vídeo", key="oraculo_youtube")
            else: fonte_dados = st.file_uploader(f"Faça o upload do arquivo .{tipo_fonte.lower()}", type=[tipo_fonte.lower()], key="oraculo_uploader")
        
        with st.expander("2. Escolha do Modelo", expanded=True):
            provedor = st.selectbox("Selecione o Provedor", list(CONFIG_MODELOS.keys()), key="oraculo_provedor")
            if provedor: modelo = st.selectbox("Selecione o Modelo", CONFIG_MODELOS[provedor]['modelos'], key="oraculo_modelo")
        
        st.divider()
        if st.button("🚀 Inicializar Oráculo", use_container_width=True, type="primary"):
            if fonte_dados and provedor and 'modelo' in locals():
                inicializar_chain(provedor, modelo, tipo_fonte, fonte_dados)
                st.rerun()
            else:
                st.warning("Por favor, forneça uma fonte de dados e selecione um modelo.")
        
        # Botões de gerenciamento usam a variável de sessão correta
        if st.session_state.get('oraculo_memoria'):
            st.divider(); st.subheader("Gerenciar Conversa")
            if st.button("🗑️ Apagar Histórico", use_container_width=True):
                st.session_state.oraculo_memoria = []; st.rerun()
            historico_texto = ""
            for tipo, conteudo in st.session_state.oraculo_memoria:
                historico_texto += f"{tipo.capitalize()}: {conteudo}\n\n---\n\n"
            st.download_button(
                label="📥 Baixar Histórico (.txt)", data=historico_texto.encode('utf-8'),
                file_name=f"historico_oraculo_{int(time.time())}.txt", mime="text/plain",
                use_container_width=True, disabled=not historico_texto
            )

# --- Execução Principal ---
construir_sidebar()
pagina_principal_chat()