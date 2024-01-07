import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.nn import functional as F
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
nlp = pipeline('text-generation', model=model_name)
print(nlp.tokenizer)
print(nlp.model.num_parameters())


def gpt2_answer(question):
    global nlp

    res = nlp(question, max_length=30)[0]
    answer = res['generated_text']
    i = 0
    try:
        i = answer.index("?")
        answer = answer[i + 1:].strip()
        try:
            try:
                i = answer.rindex(".")
            except:
                i = answer.rindex("?")
        except:
            i = answer.rindex("!")
    except:
        i = answer.rindex(".")

    if i is not None:
        answer = answer[0:i + 1].strip()

    if answer == "":
        answer = "..."

    return str(answer).strip()


if __name__ == '__main__':
    q = "Once upon a time i was at fishing"
    a = gpt2_answer(q)
    print(a)
