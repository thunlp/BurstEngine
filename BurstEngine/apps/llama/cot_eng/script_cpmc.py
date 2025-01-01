import json
import random
from typing import List


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def gen_source_name(r: random.Random):
    return r.choice(
        [
            "文章：",
            "文档：",
            "原文：",
            "输入：",
            "source: ",
            "document: ",
            "context: ",
            "input: ",
            "text: ",
            "doc: ",
            "Source: ",
            "Document: ",
            "Context: ",
            "Input: ",
            "Text: ",
            "Doc: ",
        ]
    )


def gen_question_name(r: random.Random):
    return r.choice(
        [
            "问题：",
            "问：",
            "试问：",
            "询问：",
            "question: ",
            "query: ",
            "ask: ",
            "Question: ",
            "Query: ",
            "Ask: ",
        ]
    )


def gen_choice_name(r: random.Random):
    return r.choice(["Options:\n", "Candidates:\n", "选项：\n", "候选项：\n", "选择：\n" ""])


def gen_hint_content(r: random.Random):
    return r.choice(
        [
            "回答问题",
            "根据文章中的内容回答问题",
            "问答",
            "根据文本内容回答",
            "依据文章中所说的内容，回答下列问题",
            "看完这段文字后，回答这个问题",
            "根据上下文写出答案",
        ]
    )


def gen_hint_choice(r: random.Random):
    return r.choice(
        ["选择其中一项", "选择题", "根据文章中的内容选择正确答案", "单选", "根据文本内容和问题选择答案", "多项选择", "选择你认为正确的一项", "从以上选项中选择一项答案", ""]
    )


def gen_name_answer(r: random.Random):
    return r.choice(
        [
            "答案：",
            "Answer: ",
            "answer: ",
            "Ans: ",
            "ans: ",
            "答：",
            "解：",
        ]
    )


def gen_name_reason(r: random.Random):
    return r.choice(["原因：", "理由：", "推理：", "证据：", "reason: ", "evidence: ", "thought: "])


def gen_opt_temp(r: random.Random):
    return r.choice(["({})", "[{}]", "<{}>", "{}.", "<option_{}>"])


def get_opt_name(r: random.Random):
    return r.choice([48, 65, 97])


def transform(data, num_sample: int, r: random.Random):
    inp = data["input"] + "\n" + gen_name_answer(r)
    out = data["<ans>"]
    # print ("choice\n=====\n"+inp+out)
    return {"input": inp, "output": out}
