import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.nn import functional as F


def bert_answer(question, text):
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    # question = "What is the capital of France?"
    # text = "The capital of France is Paris."
    inputs = tokenizer.encode_plus(question, text, return_tensors='pt', add_special_tokens=True, truncation=True,
                                   padding="max_length")
    start, end = model(**inputs, return_dict=False)
    start_max = torch.argmax(F.softmax(start, dim=-1))
    end_max = torch.argmax(F.softmax(end, dim=-1)) + 1  ## add one ##because of python list indexing
    answer = tokenizer.decode(inputs["input_ids"][0][start_max: end_max])

    return answer
