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

# Configuração da Página
st.set_page_config(
    page_title="Agente Especialista em Editais",
    page_icon="📋",
    layout="wide"
)

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

def criar_hash_unico(doc: Document) -> str:
    meta = dict(doc.metadata)
    meta["preview"] = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()

@st.cache_resource(show_spinner=False)
def carregar_motor_rag(api_key_val):
    if not api_key_val:
        return None, None
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key_val
    )
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.0,
        google_api_key=api_key_val
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

    batch_size = 10
    i = 0
    while i < len(trechos):
        batch_splits = trechos[i:i + batch_size]
        batch_ids = ids_trechos[i:i + batch_size]
        try:
            vstore.add_documents(batch_splits, ids=batch_ids)
            i += batch_size
            time.sleep(0.4)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(10)
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

# Barra Lateral (Upload e Gestão)
with st.sidebar:
    st.header("⚙️ Painel de Controle")
    
    if not api_key:
        st.warning("⚠️ Chave GEMINI_API_KEY não configurada nos Secrets!")
        user_key = st.text_input("Insira sua chave do Gemini temporariamente:", type="password")
        if user_key:
            api_key = user_key
            st.rerun()

    st.subheader("📄 Editais Carregados")
    pdfs_existentes = glob.glob(f"{PASTA_EDITAIS}/*.pdf")
    if pdfs_existentes:
        for p in pdfs_existentes:
            st.write(f"• `{os.path.basename(p)}`")
    else:
        st.info("Nenhum PDF na pasta ainda.")

    st.divider()
    st.subheader("📤 Enviar novo Edital")
    upload_arquivos = st.file_uploader(
        "Arraste novos editais em PDF:",
        type=["pdf"],
        accept_multiple_files=True
    )
    if upload_arquivos:
        for arq in upload_arquivos:
            caminho_salvar = os.path.join(PASTA_EDITAIS, arq.name)
            with open(caminho_salvar, "wb") as f:
                f.write(arq.getbuffer())
        st.success(f"{len(upload_arquivos)} arquivo(s) salvo(s)!")
        if st.button("🔄 Indexar Novos Editais"):
            with st.spinner("Indexando trechos no banco vetorial..."):
                qtd_arq, qtd_trechos = indexar_arquivos(vectorstore)
                st.success(f"Concluído! {qtd_trechos} trechos indexados.")
                st.rerun()

    st.divider()
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

# Área Principal de Chat
st.title("📋 Agente Especialista em Editais")
st.markdown("Consulte regras, materiais permitidos, cronogramas e proibições de vestibulares e concursos com embasamento oficial de páginas.")

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou seu assistente de editais. Faça sua pergunta sobre regras da prova, documentos permitidos, canetas, horários ou isenção de taxa."
        }
    ]

# Sugestões rápidas
cols = st.columns(4)
with cols[0]:
    if st.button("🖊️ Qual caneta levar?"):
        st.session_state.nova_pergunta = "É possível o candidato levar qual tipo de caneta no dia da prova?"
with cols[1]:
    if st.button("🪪 Documentos aceitos"):
        st.session_state.nova_pergunta = "Quais são os documentos de identificação válidos para realizar a prova?"
with cols[2]:
    if st.button("🛑 O que elimina?"):
        st.session_state.nova_pergunta = "Quais situações ou objetos causam a eliminação imediata do candidato?"
with cols[3]:
    if st.button("⏰ Horário dos portões"):
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

        # Gera resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Analisando as regras do edital..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                trechos_recuperados = retriever.invoke(pergunta_para_rodar)
                
                if not trechos_recuperados:
                    resposta = "Não encontrei informações sobre este tópico nos editais cadastrados."
                else:
                    contexto_formatado = formatar_contexto(trechos_recuperados)
                    cadeia = prompt | llm | StrOutputParser()
                    resposta = cadeia.invoke({
                        "context": contexto_formatado,
                        "question": pergunta_para_rodar
                    })
                
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
