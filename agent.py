from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
from search import EmbeddingFaiss
from prompt import ACTION_PROMPT, SEARCH_PROMPT, NORMAL_PROMPT
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import copy
import json
import utils
import os
import difflib
from datetime import datetime
from pathlib import Path

def flatten_dict(d, prefix=""):
    result = []
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix} > {k}"
        if isinstance(v, dict):
            result.extend(flatten_dict(v, prefix=key + " > "))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    result.extend(flatten_dict(item, prefix=f"{key} > {i} > "))
                else:
                    result.append(f"{key} > {i}: {item}")
        else:
            result.append(f"{key}: {v}")
    return result

EDIT_PROMPT = """
당신은 사용자에게 유용한 정보를 제공하는 AI 어시스턴트이다.
당신은 검색된 문서가 있을 경우, 그 문서를 바탕으로만 답변해야만 한다.
검색된 문서가 과거의 정보라도 가장 최신 정보이고, 실제 날짜와 무관하게 무조건 현재라고 생각해야만 한다.
문서에 존재하지 않는 정보에 대해서는 절대로 지어내거나 추론하지 말고, '모르겠습니다' 또는 '해당 정보는 문서에 없습니다'라고 대답해야 한다.
당신은 검색된 문서에 사용자의 요구에 따라 내용을 추가하거나 삭제하거나 또는 수정해야한다.
검색된 문서는 json형태로 중첩된 딕셔너리 속에 문장들의 리스트로 구성되어있다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
현재 문서 내 너의 위치는 {current_path}입니다. 반드시 현재 위치에서 가능한 **하위 키 중 하나만** `path`로 선택하십시오.
`path`는 절대 여러 키를 점(.)으로 연결하지 마십시오. 예를 들어 '인물.기본 정보'는 잘못된 path입니다. 
올바른 예: '기본 정보' 사용자의 요구에 응하기 위해 삭제/추가/수정해야 하는
문서의 위치로 가기위한 다음 path를 출력하시오.

당신이 사용할 수 있는 Action 목록과 설명은 아래와 같다.
add: 사용자가 말한 정보가 기존 문서에 없을 때 문서에 문장을 추가하는 Action이다.
delete: 사용자가 기존 문서에 있는 문장을 완전히 삭제해야 할때 사용하는 Action이다.
change: 사용자가 말한 정보가 기존 문서에 있는 내용과 달라서, 문장을 수정할때 혹은 문장의 일부만 삭제할때 사용하는 Action이다.

답변은 무조건 문서의 내용을 기반으로만 해야 한다.
문서에 없는 내용에 대해서는 답변하지 않고 'UNKNOWN'이라고만 출력한다.
지식이나 추론을 절대로 사용하지 마라.
"""

ADD_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에 문장을 추가해야 한다.
검색된 문서는 json형태로 새로운 문장을 추가할 위치는 {current_path}의 리스트의 idx이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
새로 추가할 내용을 add_text에 넣고, 그 위치를 idx로 지정해라.
"""

DELETE_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에서 문장을 삭제해야 한다.
검색된 문서는 json형태로 삭제할 문장의 위치는 {current_path}의 리스트의 idx 문장이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
삭제할 문장의 위치를 idx로 지정해라.
"""

CHANGE_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 문서에 문장을 수정해야 한다.
검색된 문서는 json형태로 문장을 수정할 위치는 {current_path}의 리스트의 idx이다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
수정이 완료된 문장을 change_text에 넣고, 그 위치를 idx로 지정해라.
"""

with open("./document.json", 'r') as f:
    document_json = json.load(f)

title2path = {title: meta["file_path"] for title, meta in document_json.items()}

def flatten_dict(d, prefix=""):
    result = []
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix} > {k}"
        if isinstance(v, dict):
            result.extend(flatten_dict(v, prefix=key + " > "))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    result.extend(flatten_dict(item, prefix=f"{key} > {i} > "))
                else:
                    result.append(f"{key} > {i}: {item}")
        else:
            result.append(f"{key}: {v}")
    return result

class ActionResponse(BaseModel):
    action: Literal["reset", "normal", "search", "edit"]

class SearchResponse(BaseModel):
    target: str

class changeResponse(BaseModel):
    idx: int
    change_text: str

class addResponse(BaseModel):
    idx: int
    add_text: str

class deleteResponse(BaseModel):
    idx: int

class BiRAGAgent():
    def __init__(self, explorer, document_dict):
        self.client = OpenAI()
        self.model = "gpt-4o"
        self.search_engine = EmbeddingFaiss(explorer, document_dict, "./DB")
        self.history = []
        self.data = None
        self.search_target = ""
        self.info = None

    def make_message(self, role, content):
        return {"role": role, "content": content}

    def call_openai(self, messages, format=None, stream=False):
        if stream:
            return self.client.chat.completions.create(model=self.model, messages=messages, stream=True)
        elif format is None:
            return self.client.chat.completions.create(model=self.model, messages=messages).choices[0].message.content
        elif isinstance(format, dict):
            return self.client.chat.completions.create(model=self.model, messages=messages, response_format=format).choices[0].message.content
        else:
            return self.client.beta.chat.completions.parse(model=self.model, messages=messages, response_format=format).choices[0].message.parsed

    def create_pathfinder_schema(self, subtitle_candidates):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pathfinder",
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "enum": list(subtitle_candidates)},
                        "action": {"type": "string", "enum": ["add", "change", "delete"]}
                    },
                    "required": ["path", "action"]
                }
            }
        }

    def resolve_path(self, edit_messages, data):
        target = data
        edit_path = []

        while isinstance(target, dict):
            current_path = "/".join(edit_path)
            path_messages = copy.deepcopy(edit_messages)
            path_messages.append(self.make_message("user", EDIT_PROMPT.format(current_path=current_path)))
            response_raw = self.call_openai(path_messages, self.create_pathfinder_schema(target.keys()))

            try:
                path_response = json.loads(response_raw)
            except json.JSONDecodeError as e:
                utils.print_log(f"[JSONDecodeError] {e} | 응답: {response_raw}")
                raise e

            next_path = path_response["path"]
            action = path_response["action"]
            if next_path not in target:
                closest = difflib.get_close_matches(next_path, target.keys(), n=1)
                next_path = closest[0] if closest else next_path

            edit_path.append(next_path)
            target = target[next_path]

        return edit_path, action

    def save_json(self, action_name="edit"):
        file_path = title2path[self.search_target]
        history_dir = Path("./history")
        history_dir.mkdir(exist_ok=True)

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self.search_target.replace(" ", "_")
            filename = f"{action_name}_{safe_title}_{timestamp}.json"
            history_path = os.path.join("./history", filename)

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(old_data, f, indent=2, ensure_ascii=False)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


    def update_embedding(self, doc_title, new_doc_data):
        file_path = title2path[doc_title]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_doc_data, f, indent=2, ensure_ascii=False)

        content = json.dumps(new_doc_data, ensure_ascii=False)
        doc = Document(page_content=content, metadata={"source": file_path})
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        new_index = FAISS.from_documents([doc], embedding_model)
        self.search_engine.vectorstore.merge_from(new_index)

    def edit_document(self):
        if not self.search_target:
            return "먼저 문서를 검색한 후에 수정할 수 있습니다."

        edit_messages = [self.make_message("system", EDIT_PROMPT)] + self.history
        edit_messages.append(self.make_message("user", f"검색된 문서의 구조는 다음과 같다. {self.info}"))

        edit_path, action = self.resolve_path(edit_messages, self.data)
        current_path = "/".join(edit_path)
        edit_messages.append(self.make_message("user", f"수행할 액션: {action} (현재 위치: {current_path})"))

        if action == "add":
            edit_messages.append(self.make_message("user", ADD_PROMPT.format(current_path=current_path)))
            add_resp = self.call_openai(edit_messages, addResponse)
            result = self.add_document(edit_path, add_resp)

        elif action == "delete":
            edit_messages.append(self.make_message("user", DELETE_PROMPT.format(current_path=current_path)))
            delete_resp = self.call_openai(edit_messages, deleteResponse)
            result = self.delete_document(edit_path, delete_resp)

        elif action == "change":
            edit_messages.append(self.make_message("user", CHANGE_PROMPT.format(current_path=current_path)))
            change_resp = self.call_openai(edit_messages, changeResponse)
            result = self.change_document(edit_path, change_resp)

        self.save_json(action_name=action)
        self.update_embedding(self.search_target, self.data)
        self.reset_history()
        return result

    def add_document(self, path_list, add_response):
        target = self.data
        for path in path_list:
            target = target[path]
        target.insert(add_response.idx, add_response.add_text)
        return "문장을 추가했습니다."

    def change_document(self, path_list, change_response):
        target = self.data
        for path in path_list[:-1]:
            target = target[path]
        last_key = path_list[-1]

        if isinstance(target[last_key], list):
            target[last_key][change_response.idx] = change_response.change_text
        else:
            # 리스트가 아닌 경우는 idx 무시하고 전체 값을 덮어쓴다
            target[last_key] = change_response.change_text

        return "문장을 변경했습니다."

    def delete_document(self, path_list, delete_response):
        target = self.data
        for path in path_list:
            target = target[path]
        del target[delete_response.idx]
        return "문장을 삭제했습니다."

    def reset_history(self):
        self.history = []

    def action_selector(self, user_input):
        self.history.append(self.make_message("user", user_input))
        messages = [self.make_message("system", ACTION_PROMPT)] + self.history
        return self.call_openai(messages, ActionResponse).action

    def search_document(self):
        messages = [self.make_message("system", SEARCH_PROMPT)] + self.history
        self.search_target = self.call_openai(messages, SearchResponse).target
        document, info, data = self.search_engine(self.search_target)
        self.data = data
        self.info = info
        
        flat_doc_lines = flatten_dict(self.data)
        flat_doc_text = "\n".join(flat_doc_lines)
        self.history.append(self.make_message("assistant", f"[검색된 문서 내용]\n{flat_doc_text}"))
        
        return self.call_openai([self.make_message("system", NORMAL_PROMPT)] + self.history, stream=True)

    def __call__(self, user_input):
        action = self.action_selector(user_input)
        if action == "reset":
            self.reset_history()
            return "대화 기록을 초기화했어요."
        elif action == "normal":
            return self.call_openai([self.make_message("system", NORMAL_PROMPT)] + self.history, stream=True)
        elif action == "search":
            return self.search_document()
        elif action == "edit":
            return self.edit_document()
