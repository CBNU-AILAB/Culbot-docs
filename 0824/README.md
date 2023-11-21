# 2023년 8월 24일 세미나

> DoctorGPT에 대해 공부하였습니다.

## score.py
원래 data의 ouput값과 모델의 output(모델의 응답)이 얼마나 유사한지 GPT4에서 점수를 측정한 코드입니다. <br> 
#### prompt 

```
        you are an AI that measures scores for predicted answers to questions as a medical expert.
        "Based on the medical facts you possess and the given question-answer set, please measure the 'output predicted by the AI model' accuracy scores  from a medical perspective on a scale of 0 to 10."

        Question: 
        `
            {d['instruction']}
        `

        Answer:
        `   
            {d['output']}
        `

        The output predicted by the AI model:
        `
            {d['pred']}
        `

        The output response only contains score.
```
## 참고자료
1. https://www.youtube.com/watch?v=J9nJh33GM-w
2. https://colab.research.google.com/github/llSourcell/DoctorGPT/blob/main/llama2.ipynb#scrollTo=-uvmvhn148bM