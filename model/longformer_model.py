import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, LongformerTokenizer, \
    LongformerForQuestionAnswering

tokenizer = LongformerTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

start = time.time()

text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this. "
question = "What has Huggingface done ?"
encoding = tokenizer(question, text, return_tensors="pt")
print(encoding)
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, return_dict=False)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

print(answer)

end = time.time()
print(end - start)
