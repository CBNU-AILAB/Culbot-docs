import openpyxl
import json

file_path = '/Users/kim-yeongsang/Desktop/instructino_ai/5. 교수님 작성 Q&A.xlsx'

# xlsx 파일 열기
workbook = openpyxl.load_workbook(file_path)

# 첫 번째 시트 선택
sheet = workbook.active

# 첫 번째 행을 건너뛰어 헤더를 제외하고 데이터를 리스트로 저장
data_list = [row for row in sheet.iter_rows(min_row=2, values_only=True)]
json_list = []
dict = {}
for i,data in enumerate(data_list):
    json_list.append({"instructino":data[2], "input":"", "output":data[3]})


# JSON 파일 저장
with open("data_medical.json", 'w') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)
# for data in json_list:
    # print(data)