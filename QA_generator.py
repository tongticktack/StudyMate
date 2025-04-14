from pydantic import BaseModel
from openai import OpenAI
from typing import Literal, List
from prompt import EDIT_PROMPT
import json
from tqdm import tqdm
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

QAGEN_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 내용을 추가하거나 삭제하거나 또는 변경하는 AI를 평가하기 위한
데이터셋을 만들어야 한다. 데이터셋은 question, request, answer, action을  key값으로 하는 json 형태이며
각 action의 목록과 데이터셋의 제작 가이드는 다음과 같다. 각 action 마다 2개씩, 한번에 여섯개의 데이터셋을 만드시오.

add: 
    사용자가 말한 정보가 기존 문서에 없을 때 문서에 내용을 추가하는 action이다.
    문서에서 답변할 수 없는 질문을 question으로 작성해야한다.
    그리고 그 question에 대한 정답을 answer로 작성한다.
    모델이 문서에 내용을 추가할 수 있도록, 해당 문서에 answer에 해당하는 내용을 추가하라고 하는 요청을 request로 작성한다.

delete: 
    사용자가 기존 문서에 있는 내용의 삭제를 요청할때 사용하는 action이다.
    문서에서 답변할 수 있는 질문을 question으로 작성해야한다.
    answer에 UNKNOWN을 작성한다.
    모델이 문서에 내용을 삭제할 수 있도록, 해당 문서에 answer에 해당하는 내용을 삭제하라고 하는 요청을 request로 작성한다.

change: 
    사용자가 말한 정보가 기존 문서에 있는 내용과 다를때 사용하는 action이다.
    문서에서 답변할 수 있는 질문을 question으로 작성해야한다.
    그리고 그 question에 대한 오답을 answer로 작성한다.
    모델이 문서에 내용을 수정할 수 있도록, 해당 문서에 answer에 해당하는 내용으로 수정하라고 하는 요청을 request로 작성한다.

예시:
    "document": "오타니 쇼헤이 신장: 193cm, 체중: 86Kg, ...",
    "question": "오타니 쇼헤이의 몸무게를 알려줘",
    "request": "오타니 쇼헤이는 최근에 벌크업을 해서 몸무게가 90kg이 됬어 수정해줘.",
    "answer": "90kg",
    "action": "change"

    "document": "오타니 쇼헤이 신장: 193cm, 체중: 86Kg.",
    "question": "오타니 쇼헤이의 국적을 알려줘",
    "request": "오타니 쇼헤이는 일본에서 태어났어.",
    "answer": "일본",
    "action": "add"

    "document": "오타니 쇼헤이 신장: 193cm, 체중: 86Kg",
    "question": "오타니 쇼헤이의 키를 알려줘",
    "request": "오타니 쇼헤이의 키에 대한 정보를 없애줘.",
    "answer": "UNKNOWN",
    "action": "delete"

"""

title2path = {
    "오타니 쇼헤이": "./DB/docs_0.json",
    "손흥민": "./DB/docs_1.json",
    "박지성": "./DB/docs_2.json",
    "크리스티아누 호날두": "/DB/docs_3.json",
    "켄 톰프슨": "/DB/docs_4.json",
    "데니스 리치": "/DB/docs_5.json",
    "크리스토퍼 콜럼버스": "/DB/docs_6.json",
    "니콜라 테슬라": "/DB/docs_7.json",
    "토마스 에디슨": "/DB/docs_8.json",
    "한강 (작가)": "/DB/docs_9.json",
    "삼성전자": "/DB/docs_10.json",
    "LG전자": "/DB/docs_11.json",
    "성균관대학교": "/DB/docs_12.json",
    "서울대학교": "/DB/docs_13.json",
    "다이제": "/DB/docs_14.json",
    "찰스 다윈": "/DB/docs_15.json",
    "마하트마 간디": "/DB/docs_16.json",
    "조지 워싱턴": "/DB/docs_17.json",
    "이이": "/DB/docs_18.json",
    "마틴 루터 킹": "/DB/docs_19.json",
    "SK 하이닉스": "/DB/docs_20.json"
}

class QAResponse(BaseModel):
    question: str
    request: str
    answer: str
    action: Literal["add", "change", "delete"]

class QAResponseSet(BaseModel):
    responses: List[QAResponse] 

    class Config:
        min_items = 6
        max_items = 6

class EmbeddingFaiss():
    def __init__(self, explorer, document_dict, dir_path):
        self.explorer = explorer
        self.document_dict = document_dict
        self.dir_path = dir_path
    
    def info_maker(self, data):
        info = ""
        for key in data.keys():
            if type(data[key]) == dict:
                info += f"{key}:"
                info += " {\n"
                for key2 in data[key].keys():
                    info += f"\t{key2}\n"
                info += "}\n"
            else:
                info += f"{key}\n"
        return info

    def __call__(self, document):
        document_path = os.path.join(self.dir_path, self.document_dict[document])
        with open(document_path, 'r') as f:
            data = json.load(f)
        info = self.info_maker(data)
        return info, data


class QAGen():
    def __init__(self, explorer, document_dict, search_target):
        self.client = OpenAI()
        self.model = "gpt-4o-2024-08-06"
        self.search_engine = EmbeddingFaiss(explorer, document_dict, "./DB")
        self.history = []
        self.search_target = search_target
        self.info, self.data = self.search_engine(search_target)

    def make_message(self, role, content):
        return {"role": role, "content": content}
    
    def gpt_agent_pydantic(self, messages, format):
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=format)
        return response.choices[0].message.parsed
    
    def answer_Q(self):
        print(self.search_target)
        answer_messages = [self.make_message("system",QAGEN_PROMPT ), 
                           self.make_message("assistant", f"{self.search_target} 문서 검색\n{str(self.data)}\n")]
        answer = self.gpt_agent_pydantic(answer_messages, QAResponseSet)
        return answer

    def save_json(self):
        file_path = title2path[self.search_target]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent="\t", ensure_ascii=False)

    def __call__(self):
        answer = self.answer_Q()
        return answer

def load_explorer(faiss_index_path, embedding_model):
    explorer = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
    return explorer

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
    "이이": "docs_18.json",
    "마틴 루터 킹": "docs_19.json",
    "SK 하이닉스": "docs_20.json"
}

valid_json = {}
valid_json['test'] = []

for document in tqdm(document_dict.keys()):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    explorer = load_explorer("./DB/faiss_index", embedding_model)
    qa_gen = QAGen(explorer, document_dict, document)
    qa_json_set = qa_gen()
    print()
    for qa_json in qa_json_set.responses:
        q_json = {}
        q_json['document'] = document
        q_json['question'] = qa_json.question
        q_json['request'] = qa_json.request
        q_json['answer'] = qa_json.answer
        q_json['action'] = qa_json.action
        print(q_json)
        valid_json['test'].append(q_json)

with open('valid.json', 'w', encoding='utf-8') as f:
    json.dump(valid_json, f, ensure_ascii=False, indent="\t")