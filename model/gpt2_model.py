import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.nn import functional as F
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "danyaljj/gpt2_question_answering_squad2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 1024
nlp = pipeline('text-generation', model=model_name, pad_token_id=tokenizer.eos_token_id,
               tokenizer=tokenizer, device=0)
print(nlp.tokenizer)
print(nlp.model.num_parameters())


def gpt2_answer(question, text):
    global nlp

    context = " ".join(text.split()[:512])
    QA_input = context + " Q: " + question + " A:"
    # greedy
    # tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 2048}
    # beam search
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 1024, 'num_beams': 5,
                             'early_stopping': True}
    res = nlp(QA_input, **tokenizer_kwargs)[0]
    answer = res['generated_text'].split(" A: ")[1]

    return str(answer).strip()


"""tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("danyaljj/gpt2_question_answering_squad2",
                                        pad_token_id=tokenizer.eos_token_id)


def gpt2_answer2(question, text):
    global tokenizer, model
    QA_input = text + " Q: " + question + " A:"
    input_ids = tokenizer.encode(QA_input, return_tensors="pt")
    outputs = model.generate(input_ids, truncation=True, padding=True, max_length=2048, num_beams=5,
                             early_stopping=True)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("Generated:", res)
    answer = res.split(" A: ")[1]

    return str(answer).strip()"""


if __name__ == '__main__':
    c = "Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s without receiving a degree, gaining practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. In 1884 he emigrated to the United States, where he became a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to develop a range of electrical and mechanical devices. His alternating current (AC) induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company eventually marketed."
    q = "When did he emigrate to United States?"
    a = gpt2_answer(q, c)
    print(a)
