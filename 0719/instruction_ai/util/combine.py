import glob
import json

# 파일을 읽어와서 데이터를 저장할 리스트
combined_data = []

# 현재 디렉토리에서 "data*.json" 파일을 찾아 리스트로 반환
file_list = glob.glob("/Users/kim-yeongsang/Desktop/instructino_ai/data_combine/data_*.json")

# 모든 파일을 반복하면서 데이터를 읽어와서 combined_data에 추가
for file_name in file_list:
    with open(file_name, 'r') as file:
        data = json.load(file)
        combined_data.extend(data)

# 합쳐진 데이터를 새로운 파일에 쓰기
with open("/Users/kim-yeongsang/Desktop/instructino_ai/data_v1.04.json", 'w') as output_file:
    json.dump(combined_data, output_file, ensure_ascii=False, indent=4)
    print(f"The result save to {output_file}")
