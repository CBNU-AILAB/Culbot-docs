import json

filename = '/Users/kim-yeongsang/Desktop/instructino_ai/data_v1.03.json'

with open(filename, 'r') as file:
    data_list = json.load(file)

max_length = -1
total_legnth = 0 
for i, data in enumerate(data_list):
    sequence = data["instruction"] + data["output"]
    total_legnth += len(sequence)
    if(max_length < len(sequence)):
        max_length = max(max_length, len(sequence))
        answer = data["instruction"] + data["output"]


print(max_length)
# print(answer)
print("average: " + str(total_legnth / len(data_list)))