import os
import glob
import time
import shutil
import hashlib
import json
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Configuração da Página - Visual de Aplicação AI de Alto Padrão
st.set_page_config(
    page_title="EditalAI • Inteligência em Vestibulares",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilização CSS Avançada (UI Design padrão Perplexity / OpenAI / Linear)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Espaçamento superior mais fluido */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 820px !important;
    }

    /* Hero Badge com animação de pulso */
    .hero-badge-container {
        display: flex;
        justify-content: center;
        margin-bottom: 14px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(168, 85, 247, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #4F46E5;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        backdrop-filter: blur(10px);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Título com Gradiente Moderno */
    .hero-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: 12px;
        color: #1E293B;
    }
    .gradient-text {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 45%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    /* Subtítulo */
    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #64748B;
        line-height: 1.55;
        max-width: 650px;
        margin: 0 auto 24px auto;
        font-weight: 400;
    }

    /* Pills das Universidades */
    .bancas-wrapper {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-bottom: 28px;
    }
    .banca-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
        transition: all 0.2s ease;
    }
    .banca-chip:hover {
        border-color: #6366F1;
        color: #4F46E5;
        background: #EEF2FF;
        transform: translateY(-1px);
    }

    /* Botões Interativos (Cards de Atalho) */
    .stButton > button {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #1E293B !important;
        border-radius: 14px !important;
        padding: 12px 14px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: center !important;
        min-height: 48px !important;
    }
    .stButton > button:hover {
        border-color: #6366F1 !important;
        color: #4F46E5 !important;
        background: #F8FAFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.12) !important;
    }

    /* Mensagens do Chat */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border: 1px solid #F1F5F9 !important;
        border-radius: 18px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
        margin-bottom: 12px !important;
        padding: 18px !important;
    }
    
    /* Input de Chat Fixo */
    [data-testid="stChatInput"] {
        border-radius: 9999px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04) !important;
    }
</style>
""", unsafe_allow_html=True)

PASTA_EDITAIS = "editais"
PASTA_DB = "chroma_db_streamlit"
os.makedirs(PASTA_EDITAIS, exist_ok=True)

# 1. Obtenção da Chave da API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

def criar_hash_unico(doc: Document) -> str:
    meta = dict(doc.metadata)
    meta["preview"] = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def carregar_motor_rag(api_key_val):
    if not api_key_val:
        return None, None
    
    os.environ["GEMINI_API_KEY"] = api_key_val
    os.environ["GOOGLE_API_KEY"] = api_key_val
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key_val
    )
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.0,
        google_api_key=api_key_val,
        max_retries=5
    )
    vstore = Chroma(
        collection_name="colecao_editais_st",
        embedding_function=embeddings,
        persist_directory=PASTA_DB
    )
    return vstore, llm_model

vectorstore, llm = carregar_motor_rag(api_key)

def indexar_arquivos(vstore):
    arquivos_pdf = glob.glob(f"{PASTA_EDITAIS}/*.pdf") + glob.glob("*.pdf")
    arquivos_pdf = list(set([os.path.abspath(f) for f in arquivos_pdf if not f.endswith("_db")]))
    
    if not arquivos_pdf:
        return 0, 0

    docs_totais = []
    for pdf in arquivos_pdf:
        try:
            loader = PyPDFLoader(pdf)
            docs_totais.extend(loader.load())
        except Exception as e:
            st.error(f"Erro ao carregar {os.path.basename(pdf)}: {e}")

    if not docs_totais:
        return len(arquivos_pdf), 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    trechos = text_splitter.split_documents(docs_totais)
    ids_trechos = [criar_hash_unico(t) for t in trechos]

    # Otimização: lotes de 50 para economizar a cota diária de requisições do Google
    batch_size = 50
    i = 0
    while i < len(trechos):
        batch_splits = trechos[i:i + batch_size]
        batch_ids = ids_trechos[i:i + batch_size]
        try:
            vstore.add_documents(batch_splits, ids=batch_ids)
            i += batch_size
            time.sleep(1.0)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(15)
            else:
                raise e

    return len(arquivos_pdf), len(trechos)

template_edital = """Você é um assistente especialista em analisar Editais de Vestibulares e Concursos Públicos.
Sua missão é responder à dúvida do usuário com base EXCLUSIVAMENTE nas regras, prazos e determinações expressas nos trechos do edital fornecidos abaixo.

DIRETRIZES FUNDAMENTAIS:
1. Responda de forma direta, clara e objetiva em português do Brasil.
2. Seja específico sobre materiais, horários, tipos de caneta, documentos aceitos e procedimentos.
3. Se o edital contiver restrições (ex: proibição de caneta com corpo escuro, lápis, relógio), mencione-as explicitamente.
4. NUNCA invente informações nem faça suposições baseadas em outros concursos. Se a resposta não estiver descrita no texto fornecido, afirme claramente: 'Essa informação não consta expressamente nos trechos analisados do edital.'
5. Sempre termine indicando a(s) página(s) e o(s) arquivo(s) de onde a resposta foi extraída.

TRECHOS EXTRAÍDOS DO EDITAL:
{context}

DÚVIDA DO CANDIDATO:
{question}

ESTRUTURA DA RESPOSTA:
**Resposta:** [Explicação clara e objetiva da regra]

**Referência no Edital:** [Nome do Arquivo e Página(s) citadas]
"""
prompt = ChatPromptTemplate.from_template(template_edital)

def formatar_contexto(documentos):
    blocos = []
    for doc in documentos:
        num_pagina = doc.metadata.get("page", 0) + 1
        arquivo_origem = os.path.basename(doc.metadata.get("source", "Edital"))
        blocos.append(f"--- [ARQUIVO: {arquivo_origem} | PÁGINA: {num_pagina}] ---\n{doc.page_content}")
    return "\n\n".join(blocos)

# Barra Lateral Elegante (Inicia recolhida para o usuário)
with st.sidebar:
    st.title("📚 Editais Disponíveis")
    
    pdfs_existentes = glob.glob(f"{PASTA_EDITAIS}/*.pdf") + glob.glob("*.pdf")
    pdfs_existentes = list(set([p for p in pdfs_existentes if not p.endswith("_db") and os.path.isfile(p)]))
    
    if pdfs_existentes:
        for p in sorted(pdfs_existentes):
            nome_limpo = os.path.basename(p).replace(".pdf", "").replace("_EditaldeInscricao", "").replace("_EditaldeInscricaoRetificado", "")
            st.markdown(f"• **{nome_limpo}**")
    else:
        st.info("Nenhum edital cadastrado no momento.")

    st.divider()
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Área de Gestão protegida dentro de um Expander recolhido
    with st.expander("⚙️ Área Administrativa"):
        st.caption("Apenas para administradores:")
        upload_arquivos = st.file_uploader(
            "Adicionar novo PDF:",
            type=["pdf"],
            accept_multiple_files=True
        )
        if upload_arquivos:
            for arq in upload_arquivos:
                caminho_salvar = os.path.join(PASTA_EDITAIS, arq.name)
                with open(caminho_salvar, "wb") as f:
                    f.write(arq.getbuffer())
            st.success(f"{len(upload_arquivos)} arquivo(s) salvo(s)!")
            if st.button("🔄 Indexar Novos Arquivos"):
                with st.spinner("Indexando..."):
                    indexar_arquivos(vectorstore)
                    st.success("Concluído!")
                    st.rerun()

# Área Principal - Hero & Branding de Nível Comercial
st.markdown("""
<div class="hero-badge-container">
    <div class="hero-badge">
        <span class="pulse-dot"></span> Base Oficial Multiversidades • 2026/2027
    </div>
</div>
<h1 class="hero-title">Consultor Inteligente de <span class="gradient-text">Editais de Vestibular</span></h1>
<p class="hero-subtitle">Consulte regras, canetas permitidas, cronogramas e critérios eliminatórios com respostas embasadas nas páginas oficiais das bancas.</p>
<div class="bancas-wrapper">
    <span class="banca-chip">🏛️ UECE</span>
    <span class="banca-chip">🎓 UNIFOR</span>
    <span class="banca-chip">📘 ENEM</span>
    <span class="banca-chip">🔬 UNICAMP</span>
    <span class="banca-chip">🏛️ UEMA</span>
    <span class="banca-chip">🏫 URCA</span>
    <span class="banca-chip">🩺 UNICHRISTUS</span>
</div>
""", unsafe_allow_html=True)

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Olá! Sou seu **Assistente de Editais**. Selecione um dos atalhos abaixo ou faça qualquer pergunta sobre canetas, materiais, horários ou documentos."
        }
    ]

# Sugestões rápidas em cards interativos
cols = st.columns(4)
with cols[0]:
    if st.button("🖊️ Tipo de Caneta", use_container_width=True):
        st.session_state.nova_pergunta = "É possível o candidato levar qual tipo de caneta no dia da prova?"
with cols[1]:
    if st.button("🪪 Documentos Aceitos", use_container_width=True):
        st.session_state.nova_pergunta = "Quais são os documentos de identificação válidos para realizar a prova?"
with cols[2]:
    if st.button("🛑 O que Elimina?", use_container_width=True):
        st.session_state.nova_pergunta = "Quais situações ou objetos causam a eliminação imediata do candidato?"
with cols[3]:
    if st.button("⏰ Horário dos Portões", use_container_width=True):
        st.session_state.nova_pergunta = "Qual é o horário limite para fechamento dos portões nos dias de prova?"

# Exibe histórico do chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Processa pergunta do usuário
pergunta_para_rodar = None
if "nova_pergunta" in st.session_state and st.session_state.nova_pergunta:
    pergunta_para_rodar = st.session_state.nova_pergunta
    st.session_state.nova_pergunta = None

input_chat = st.chat_input("Digite sua dúvida sobre o edital...")
if input_chat:
    pergunta_para_rodar = input_chat

if pergunta_para_rodar:
    if not api_key or not vectorstore:
        st.error("Configure sua chave da API do Gemini para continuar.")
    else:
        # Registra a mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": pergunta_para_rodar})
        with st.chat_message("user"):
            st.markdown(pergunta_para_rodar)

        # Gera resposta do assistente com proteção contra limite de requisições
        with st.chat_message("assistant"):
            with st.spinner("Analisando as regras do edital..."):
                trechos_recuperados = []
                erro_busca = None
                
                # Tenta até 3 vezes com intervalo caso a API do Google esteja em resfriamento após a indexação
                for tentativa in range(3):
                    try:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
                        trechos_recuperados = retriever.invoke(pergunta_para_rodar)
                        erro_busca = None
                        break
                    except Exception as e:
                        erro_busca = e
                        time.sleep(3)
                
                if erro_busca:
                    resposta = f"⚠️ **Erro na busca de trechos:**\n\n```text\n{str(erro_busca)}\n```"
                elif not trechos_recuperados:
                    resposta = "Não encontrei informações sobre este tópico nos editais cadastrados."
                else:
                    contexto_formatado = formatar_contexto(trechos_recuperados)
                    cadeia = prompt | llm | StrOutputParser()
                    try:
                        resposta = cadeia.invoke({
                            "context": contexto_formatado,
                            "question": pergunta_para_rodar
                        })
                    except Exception as e:
                        resposta = f"⚠️ Ocorreu um erro ao gerar a resposta: {e}. Tente novamente em instantes."
                
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
