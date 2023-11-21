from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json


url = "https://www.univ100.kr/qna/122"

browser = webdriver.Safari()
browser.get(url)
time.sleep(2)
# 스크롤을 이용해 페이지 로딩
SCROLL_PAUSE_TIME = 8
cnt = 5
# Get scroll height
last_height = browser.execute_script("return document.body.scrollHeight")
data_json = []
set_q= set()
now = 0
while True:
    now +=1
    print(now)
    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

    qna_elements = browser.find_elements(By.CLASS_NAME, 'list__item')  # 각 Q&A 항목을 식별하는 클래스 이름 사용

    for qna_element in qna_elements:
        question_title = qna_element.find_element(By.CLASS_NAME, 'qna__question-title-inner').text
        question_inner = qna_element.find_element(By.CLASS_NAME, 'qna__text.qna__question-text').text
        try:
            answer_element = qna_element.find_element(By.CSS_SELECTOR, '.qna__text.qna__answer-text')
            answer = answer_element.text
        except:
            answer = ""
        question = question_title + ' ' + question_inner
        if(question not in set_q):
            set_q.add(question)
            data_json.append({"instruction": question, "input":"", "output":answer})

with open("data_crawling3.json", 'w') as json_file:
    json.dump(data_json, json_file, ensure_ascii=False, indent=4)

# 웹 드라이버 종료
browser.quit()