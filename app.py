import streamlit as st
from agent import BiRAGAgent
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import utils
import time
import json
import os
import shutil
from pathlib import Path

st.set_page_config(layout="wide")

# 문서 정보 로드
with open("./document.json", 'r') as f:
    document_json = json.load(f)

document_dict = {
    doc: meta['file_name'] for doc, meta in document_json.items()
}

def load_explorer(faiss_index_path, embedding_model):
    return FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

@st.cache_resource
def get_agent():
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    explorer = load_explorer("./DB/faiss_index", embedding_model)
    return BiRAGAgent(explorer, document_dict)

agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 사이드바: 복원 기능 UI
with st.sidebar:
    st.subheader("되돌리기 기능")
    selected_doc = st.selectbox("문서 선택", list(document_dict.keys()))

    def get_history_files(doc_key):
        history_dir = Path("./history")
        return sorted([
            f for f in os.listdir(history_dir)
            if doc_key.replace(" ", "_") in f and f.endswith(".json")
        ], reverse=True)

    def restore_history_file(file_name, doc_key):
        history_path = Path("./history") / file_name
        target_path = Path("./DB") / document_dict[doc_key]
        with open(history_path, 'r', encoding='utf-8') as src, open(target_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        agent.reset_history()
        target_path = Path("./DB") / document_dict[doc_key]
        shutil.copy(history_path, target_path)

    history_files = get_history_files(selected_doc)
    if history_files:
        selected_backup = st.selectbox("복원할 백업 파일", history_files)
        if st.button("복원하기"):
            restore_history_file(selected_backup, selected_doc)
            st.success(f"{selected_backup} 복원 완료")
    else:
        st.info("이 문서는 아직 백업 기록이 없습니다.")

# ✅ 본문 채팅 UI
st.markdown("""
    <h1 style='font-size:30px;'>Read and Write RAG: rwRAG</h1>
    <hr style='margin-top:0;'>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])

if prompt := st.chat_input(placeholder="입력"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    utils.format_and_print_user_input(prompt)
    response = agent(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if type(response) == str:
            utils.print_log("Received string response")
            full_response += response + " "
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
        else:
            utils.print_log("Received stream response")
            for chunk in response:
                if isinstance(chunk, str):
                    full_response += chunk
                    time.sleep(0.05)
                elif chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

        utils.format_and_print_genai_response(full_response)
        agent.history.append(agent.make_message("assistant", full_response))
        st.session_state.messages.append({"role": "assistant", "content": full_response})
