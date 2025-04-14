from pydantic import BaseModel
from openai import OpenAI
from typing import Literal
import copy
import json
import os

ANSWER_PROMPT = """
당신은 사용자에게 유용한 정보를 제공하는 AI 어시스턴트이다.
당신은 사용자의 질문에 대해 문서의 내용을 보고 답변을 해야한다.
답변은 무조건 문서에 있는 내용으로 답해야 한다.
문서에 없는 내용을 답변으로 할 수 없으며, 문서에서 답을 찾을 수 없을 경우 답변으로 UNKNOWN을 출력해라.
답변은 답에 해당하는 한 단어 또는 단어 리스트를 출력한다.
<예시>
    <문서>
    일론 머스크
    본명: 일론 리브 머스크
    출생: 1973년 7월 3일
    국적: 남아프리카 공화국, 
        캐나다, 
        미국
    부모: 메이 머스크(어머니),
        데미안 머스크(아버지)
    <문서 끝>

    <질문&답변>
    질문: 일론 머스크의 출생일은? 
    답변: 1973년 7월 3일

    질문: 일론 머스크의 생일은? 
    답변: 7월 3일

    질문: 일론 머스크의 출생년도는? 
    답변: 1973년

    질문: 일론 머스크의 아버지는?
    답변: 데미안 머스크

    질문: 일론 머스크의 국적은?
    답변: 남아프리카 공화국, 캐나다, 미국

    질문: 일론 머스크의 배우자는?
    답변: UNKNOWN

    질문: 일론 머스크의 회사는?
    답변: UNKNOWN
    <질문&답변 끝>
<예시 끝>
"""

ACTION_PROMPT = """
사용자의 입력에 알맞는 Action을 골라야한다.
Action의 목록과 설명은 아래와 같다.
reset: assistant와의 대화 기록을 삭제한다.
QA: 사용자의 질문에 대해 답을 한다.
edit: 검색된 외부지식의 내용을 수정한다.
"""

EDIT_PROMPT = """
당신은 검색된 문서에 사용자의 요구에 따라 내용을 추가하거나 삭제하거나 또는 수정해야한다.
검색된 문서는 json형태로 중첩된 딕셔너리 속에 문장들의 리스트로 구성되어있다.
문장 리스트 안의 index는 0부터 시작한다. 리스트의 첫번째 문장은 idx가 0이고, 두번째 문장은 idx가 1이다.
현재 너의 문장안에서의 위치는 {current_path}이다. 사용자의 요구에 응하기 위해 삭제/추가/수정해야 하는
문서의 위치로 가기위한 다음 path를 출력하시오.

당신이 사용할 수 있는 Action 목록과 설명은 아래와 같다.
add: 사용자가 말한 정보가 기존 문서에 없을 때 문서에 문장을 추가하는 Action이다.
delete: 사용자가 기존 문서에 있는 문장을 완전히 삭제해야 할때 사용하는 Action이다.
change: 사용자가 말한 정보가 기존 문서에 있는 내용과 달라서, 문장을 수정할때 혹은 문장의 일부만 삭제할때 사용하는 Action이다.
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

title2path = {}
for document in document_json.keys():
    title2path[document] = document_json[document]['file_path'] 

class ActionResponse(BaseModel):
    action: Literal["reset", "QA", "edit",]

class changeResponse(BaseModel):
    idx: int
    change_text: str

class addResponse(BaseModel):
    idx: int
    add_text: str

class deleteResponse(BaseModel):
    idx: int

class QAResponse(BaseModel):
    answer: str

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


class BiRAGAgent():
    def __init__(self, explorer, document_dict, search_target):
        self.client = OpenAI()
        # self.model = "gpt-4o-2024-08-06"
        self.model ="gpt-4o-mini"
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
    
    def gpt_agent_json(self, messages, format):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=format)
        
        return response.choices[0].message.content
    
    def save_json(self):
        file_path = title2path[self.search_target]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent="\t", ensure_ascii=False)

    def reset_history(self):
        self.history = []
        return
    
    def action_selector(self, user_input):
        user_message = self.make_message("user", user_input)
        self.history.append(user_message)
        action_messages = [self.make_message("system", ACTION_PROMPT)]
        for history in self.history:
            action_messages.append(history)
        response = self.gpt_agent_pydantic(action_messages, ActionResponse)
        action_type = response.action
        return action_type
    
    def answer_Q(self):
        self.history.append(self.make_message("assistant", f"{self.search_target} 문서 검색\n{str(self.data)}\n"))
        answer_messages = [self.make_message("system",ANSWER_PROMPT )]
        for history in self.history:
            answer_messages.append(history)
        answer = self.gpt_agent_pydantic(answer_messages, QAResponse).answer
        self.history.append(self.make_message("assistant", answer))
        return answer

    def create_pathfinder_schema(self, next_path_candidates):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "pathfinder",
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "현재 위치에서 문서에서 수정/삭제/추가를 \
                                  해야하는 위치에 도달하기 위한 다음 경로",
                            "enum": list(next_path_candidates) 
                        },
                        "action": {
                            "type": "string",
                            "description": "문서에서 수정/삭제/추가 중 무엇을 할 지 정하는 변수",
                            "enum": ["add", "change", "delete"]
                        }
                    },
                    "required": ["path", "action"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        return response_format

    def edit_document(self):
        edit_messages = [self.make_message("system", EDIT_PROMPT)]
        for history in self.history:
            edit_messages.append(history)
        edit_messages.append(self.make_message("user", f"검색된 문서의 구조는 다음과 같다. {self.info}"))

        edit_path = []
        target = self.data
        while True:
            if type(target) == list:
                break
            subtitle_candidates = target.keys()
            path_messages = copy.deepcopy(edit_messages)
            current_path = "/".join(edit_path)
            path_messages.append(self.make_message("user", EDIT_PROMPT.format(current_path=current_path)))
            path_response = self.gpt_agent_json(path_messages, self.create_pathfinder_schema(subtitle_candidates))
            print(path_response)
            path_response = json.loads(path_response)
            edit_path.append(path_response["path"])
            target = target[path_response["path"]]

        current_path = "/".join(edit_path)
        edit_messages.append(self.make_message("user", f"너의 현재 위치는 {current_path}, 수행해야할 액션은 {path_response['action']}이다."))
        
        if path_response["action"] == "add":
            edit_messages.append(self.make_message("user", ADD_PROMPT.format(current_path=current_path)))
            add_response = self.gpt_agent_pydantic(edit_messages, addResponse)
            print(add_response)
            response =  self.add_document(edit_path, add_response)

        elif path_response["action"] == "delete":
            edit_messages.append(self.make_message("user", DELETE_PROMPT.format(current_path=current_path)))
            delete_response = self.gpt_agent_pydantic(edit_messages, deleteResponse)
            print(delete_response)
            response = self.delete_document(edit_path, delete_response)

        elif path_response["action"] == "change":
            edit_messages.append(self.make_message("user", CHANGE_PROMPT.format(current_path=current_path)))
            change_response = self.gpt_agent_pydantic(edit_messages, changeResponse)
            print(change_response)
            response = self.change_document(edit_path, change_response)
        
        self.save_json()
        return response
    
    def add_document(self, edit_path, add_response):
        path_list = edit_path
        add_text = add_response.add_text
        idx = add_response.idx
        target = self.data
        for path in path_list:
            target = target[path]
        if idx >= len(target):
            target.append(add_text)
        else:
            target.insert(idx, add_text)

        return "내용을 추가했습니다."
    
    def change_document(self, edit_path, change_response):
        path_list = edit_path
        idx = change_response.idx
        change_text = change_response.change_text
        target = self.data
        for path in path_list:
            target = target[path]
        target[idx] = change_text

        return "내용을 변경했습니다."
    
    def delete_document(self, edit_path, delete_response):
        path_list = edit_path
        idx = delete_response.idx
        target = self.data
        for path in path_list:
            target = target[path]
        del target[idx]

        return "내용을 삭제했습니다."

    def __call__(self, user_input):
        action_type = self.action_selector(user_input)
        if action_type == "reset":
            self.reset_history()
            return "대화 기록이 삭제되었습니다."
        elif action_type == "QA":
            answer = self.answer_Q()
            return answer
        elif action_type == "edit":
            return self.edit_document()

        