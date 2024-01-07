import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch.nn import functional as F

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# bert-large-uncased-whole-word-masking-finetuned-squad
# valhalla/longformer-base-4096-finetuned-squadv1
# deepset/bert-large-uncased-whole-word-masking-squad2
# deepset/roberta-base-squad2
# deepset/roberta-large-squad2 *
# patrickvonplaten/opt_metaseq_125m
# deepset/xlm-roberta-base-squad2
# model_name = "C:/Users/Bogdan/PycharmProjects/Historical Conversational Agent/fine_tune/test-adversarial_qa-trained-1epoch_v2"
model_name = "deepset/roberta-large-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)
print(nlp.tokenizer)
print(nlp.model.num_parameters())


def bert_answer(question, text):
    global nlp

    if nlp is None:
        model_name = "deepset/roberta-large-squad2"
        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)

    QA_input = {
        'question': question,
        'context': text
    }
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512, 'return_tensors': 'pt'}
    res = nlp(QA_input, **tokenizer_kwargs)
    # print(res)

    answer = res['answer']

    return answer


if __name__ == '__main__':
    c = "John has 3 apples."
    q = "How many apples does John have?"
    a = bert_answer(q, c)
    print("a: " + a)
