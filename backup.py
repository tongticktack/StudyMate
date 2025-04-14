import os
import json

backup_dir = "./document_backup"
db_dir = "./DB"
backup_documents = os.listdir(backup_dir)
for document in backup_documents:
    document_backup_path = os.path.join(backup_dir, document)
    target_path = os.path.join(db_dir, document)
    with open(document_backup_path, 'r') as f:
        document_json = json.load(f)
    with open(target_path, 'w') as f:
        json.dump(document_json, f, ensure_ascii=False, indent="\t")