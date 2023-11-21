import re
import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup
import os

## 충북소개
# 442~453
# 456~462
# 국경일

## 도정소식
# 연도별 주요행
# num = 0
# for i in range(456, 463, 1):
#     num = num + 1
#     ""
#     url = f"""https://www.chungbuk.go.kr/www/selectBbsNttView.do?key=429&bbsNo=65&nttNo=269193&searchCtgry=&searchCnd=all&searchKrwd=&pageIndex=1"""
#     response = requests.get(url)
#
#     if response.status_code == 200:
#         html = response.text
#         soup = BeautifulSoup(html, 'html.parser')
#         title = soup.select_one("#wrapper > div.sub_visual > div > div > div.slogan > h2").get_text()
#         content = soup.select_one('#colgroup > article').get_text()
#         af_title = re.sub(r"\s+", " ", title)
#         af_content = re.sub(r"\s+", " ", content)
#         af_content = re.sub(r"파일 다운로드", " ", af_content)
#         af_content = re.sub("이미지 확대보기", " ", af_content)
#         af_content = re.sub("충청북도 상징물은 개인, 단체, 기업의 영리목적 사용을 금지하고 있습니다.", " ", af_content)
#         result = f"""
#         {af_title}\n
#         {af_content}
#         """
#         file_name_format = "information_{}.txt"
#
#         try:
#             file_name = file_name_format.format(num)
#             file_path = os.path.join("./info", file_name)
#             if not os.path.exists(file_path):
#                 f = open(file_path, "w+", encoding="utf-8")
#                 f.write("%s\n" % result)
#                 f.close
#                 print(f"The result save to {file_path}")
#                 # break
#
#         except:
#             print("failed save file")
#
#     else:
#         print(response.status_code)


url = "https://www.chungbuk.go.kr/www/selectBbsNttList.do?key=429&bbsNo=65&searchCtgry=&pageUnit=10&searchCnd=all&searchKrwd=&pageIndex=1"
browser = webdriver.Safari()
browser.get(url)
time.sleep(2)

num = 0
for page in range(0, 10):
    if page != 0:
        click_page = browser.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/main/article/div/div/div[2]/div/span[2]/a['+str(page)+']')
        browser.execute_script("arguments[0].click();", click_page)
        print("page: ", page)
        time.sleep(5)

    for i in range(1, 11):
        print("item: ", i)
        item = '/html/body/div[2]/div[2]/div/main/article/div/div/table/tbody/tr[' + str(i) + "]/td[2]/a"
        click_list = browser.find_element(By.XPATH, item)
        browser.execute_script("arguments[0].click();", click_list)
        time.sleep(5)
        result = browser.find_element(By.XPATH,
                                       '/html/body/div[2]/div[2]/div/main/article/div/div/table/tbody/tr[2]/td').text
        file_name_format = "information_{}.txt"
        try:
            file_name = file_name_format.format(num)
            file_path = os.path.join("./info", file_name)
            if not os.path.exists(file_path):
                f = open(file_path, "w+", encoding="utf-8")
                f.write("%s\n" % result)
                f.close
                num = num + 1
                print(f"The result save to {file_path}")
                # break
        except:
            print("failed save file")
        browser.back()
