from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from agent_valid import BiRAGAgent
import pandas as pd
import json
import shutil
import os

# 문서 복원: document_backup/ → DB/
backup_dir = "./document_backup"
db_dir = "./DB"

for filename in os.listdir(backup_dir):
    if filename.startswith("docs_") and filename.endswith(".json"):
        src = os.path.join(backup_dir, filename)
        dst = os.path.join(db_dir, filename)
        shutil.copyfile(src, dst)

# 평가용 데이터 로드
with open("valid.json", 'r', encoding="utf-8") as f:
    valid_dataset = json.load(f)

# 임베딩 및 벡터 검색기 로드
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
explorer = FAISS.load_local("./DB/faiss_index", embedding_model, allow_dangerous_deserialization=True)
# 문서 제목 → 파일 이름 매핑
document_dict = {
    "오타니 쇼헤이": "docs_0.json",
    "손흥민": "docs_1.json",
    "박지성": "docs_2.json",
    "크리스티아누 호날두": "docs_3.json",
    "켄 톰프슨": "docs_4.json",
    "데니스 리치": "docs_5.json",
    "크리스토퍼 콜럼버스": "docs_6.json",
    "니콜라 테슬라": "docs_7.json",
    "토마스 에디슨": "docs_8.json",
    "한강 (작가)": "docs_9.json",
    "삼성전자": "docs_10.json",
    "LG전자": "docs_11.json",
    "성균관대학교": "docs_12.json",
    "서울대학교": "docs_13.json",
    "다이제": "docs_14.json",
    "찰스 다윈": "docs_15.json",
    "마하트마 간디": "docs_16.json",
    "조지 워싱턴": "docs_17.json",
    "마틴 루터 킹": "docs_18.json",
    "SK 하이닉스": "docs_19.json"
}

# 결과 저장 리스트
first_answer_list = []
answer_list = []
predict_list = []

# 평가 시작
for data in valid_dataset['test']:
    document = data['document']
    question = data['question']
    request = data['request']
    answer = data['answer']

    # 평가용 에이전트 생성
    agent = BiRAGAgent(explorer, document_dict, document)

    try:
        # QA 수행 전 정답 확인
        first_answer = agent(question)
        print(f"first_answer: {first_answer}")

        # 문서 편집 수행
        print(agent(request))

        # 대화 이력 초기화 후 다시 질문
        agent.reset_history()
        predict = agent(question)

    except Exception as e:
        first_answer = "halted"
        predict = "halted"
        print(f"ERROR: {e}")

    first_answer_list.append(first_answer)
    answer_list.append(answer)
    predict_list.append(predict)

    print(f"predict: {predict}")
    print(f"answer: {answer}")
    print("정답" if answer.lower() == predict.lower() else "오답")

    # 편집 후 원본 복원
    for filename in os.listdir(backup_dir):
        if filename.startswith("docs_") and filename.endswith(".json"):
            src = os.path.join(backup_dir, filename)
            dst = os.path.join(db_dir, filename)
            shutil.copyfile(src, dst)

# 결과 저장
df = pd.DataFrame({
    "first_answer": first_answer_list,
    "answer": answer_list,
    "predict": predict_list
})

df.to_csv("result_eval.tsv", sep="\t", index=False)
