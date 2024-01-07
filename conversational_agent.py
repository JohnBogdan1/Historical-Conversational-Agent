import random
from get_data import *
from model.bert_model_3 import bert_answer
# from model.opt_model1 import opt_answer
# from model.gpt2_model import gpt2_answer
from model.gpt2_gen import gpt2_answer


class ConversationalAgent(object):
    def __init__(self, name, biography):
        self.name = name
        self.biography = biography

    def get_response(self, message):
        answer = bert_answer(message, self.biography)
        return answer

    def get_intent_response(self, message):
        answer = bert_answer(message, self.biography)
        return answer

    def get_gen_response(self, message):
        answer = gpt2_answer(message)
        return answer.replace(message, "")
