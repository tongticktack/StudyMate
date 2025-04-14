import os
import json

class EmbeddingFaiss():
    def __init__(self, explorer, document_dict, dir_path):
        self.explorer = explorer
        self.document_dict = document_dict
        self.dir_path = dir_path
        self.vectorstore = explorer  # ★ 이거 추가해야 `.vectorstore` 접근 가능해짐

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
        
    def __call__(self, argument):
        document = self.explorer.similarity_search(argument, k=1)[0].page_content
        document_path = os.path.join(self.dir_path, self.document_dict[document])
        print(document_path)
        with open(document_path, 'r') as f:
            data = json.load(f)
        info = self.info_maker(data)
        print(data)
        return document, info, data