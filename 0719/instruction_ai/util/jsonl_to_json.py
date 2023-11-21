import json

jsonl_file_path = '/Users/kim-yeongsang/Desktop/instructino_ai/kullm-v2.jsonl'
json_file_path = '/Users/kim-yeongsang/Desktop/instructino_ai/kullm-v2.json'

json_objects = []

# JSONL 파일 읽기
with open(jsonl_file_path, 'r') as jsonl_file:
    # 각 줄을 파싱하여 JSON 개체로 변환 후 리스트에 추가
    for line in jsonl_file:
        json_obj = json.loads(line)
        json_objects.append({"instruction":json_obj['instruction'], "input":'','output':json_obj['output']})

# JSON 파일 저장
with open(json_file_path, 'w') as json_file:
    json.dump(json_objects, json_file, ensure_ascii=False, indent=4)
