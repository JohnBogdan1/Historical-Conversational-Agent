from get_data import *
from model.bert_model_3 import bert_answer
from model.longformer_model_1 import longformer_answer


def main():
    agent_name = 'nikola tesla'
    agent_name, sentences, page_content = read_data_from_wiki(agent_name)
    print("# of sentences=", len(sentences))
    lens = [len(sentence.split(" ")) for sentence in sentences]
    print("# of words on average", sum(lens) / len(sentences))
    print("# of characters=", len(page_content))

    questions = ["Who was Nikola Tesla?", "What was Nikola Tesla best known for?", "Where were Nikola Tesla born?",
                 "When did Nikola Tesla die?", "Where did Nikola Tesla die?",
                 "How many patents did Tesla receive?", "When did Tesla leave Gospić?",
                 "When did Tesla move to Budapest?", "When did Tesla begin to work at Edison Machine Works?",
                 "How many languages did Tesla speak?", "How tall was Tesla?"]
    for question in questions:
        print("Q:", question)
        answer = bert_answer(question, page_content)
        print("A:", answer)
        print()

    # print(page_content)


if __name__ == '__main__':
    main()
