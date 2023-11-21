import json


filename = '/Users/kim-yeongsang/Desktop/instructino_ai/ko_cleaned.json'

with open(filename, 'r') as file:
    data_list = json.load(file)

convert_list = []
for i, data in enumerate(data_list):
    convert_list.append({"id": i, "tag":"multiturn", "conversations" : data['conversations'] })


# JSON 파일 저장
with open("data_multiturn.json", 'w') as json_file:
    json.dump(convert_list, json_file, ensure_ascii=False, indent=4)