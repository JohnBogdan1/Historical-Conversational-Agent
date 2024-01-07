import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.nn import functional as F
from transformers import GPT2Tokenizer, OPTForCausalLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "anas-awadalla/opt-125m-squad"
nlp = pipeline('text-generation', model=model_name, device=0)
print(nlp.tokenizer)
print(nlp.model.num_parameters())

def opt_answer(question, text):
    global nlp

    QA_input = text + "\nQuestion:" + question + "\nAnswer:"
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 2048}
    res = nlp(QA_input, **tokenizer_kwargs)[0]
    answer = res['generated_text'].split("\nAnswer:")[1]

    return answer


if __name__ == '__main__':
    c = "John has 3 apples."
    q = "How many apples does John have?"
    a = opt_answer(q, c)
    print(a)
