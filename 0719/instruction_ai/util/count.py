import json

filename = '/Users/kim-yeongsang/Desktop/instructino_ai/kullm-v2.json'

with open(filename, 'r') as file:
    data = json.load(file)
    row_count = len(data)

print("행의 수:", row_count)
