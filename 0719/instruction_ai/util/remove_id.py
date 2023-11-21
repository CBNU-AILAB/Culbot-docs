import glob
import json

filename = '/Users/kim-yeongsang/Desktop/instructino_ai/data_cbnu_v1.01_id.json'


add_id_data = []
with open(filename, 'r') as file:
    data_list = json.load(file)
    for i , data in enumerate(data_list):
        add_id_data.append({"instruction": data['instruction'], "input": data['input'], "output": data['output']})


with open("data_cbnu_v1.02.json", 'w') as output_file:
    json.dump(add_id_data, output_file, ensure_ascii=False, indent=4)
    print(f"The result save to {output_file}")

