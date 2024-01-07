import numpy as np
import nltk
from sacrebleu.metrics import BLEU
from bert_score import BERTScorer
from get_data import *
from model.bert_model_3 import bert_answer
# from model.opt_model1 import opt_answer
# from model.gpt2_model import gpt2_answer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

"""nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')"""

PATH = "data/only_summary/"
ABRAHAM_LINCOLN = 'abraham lincoln'
ALBERT_EINSTEIN = 'albert einstein'
ISAAC_ASIMOV = 'isaac asimov'
NIKOLA_TESLA = 'nikola tesla'

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def readlines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        pairs = [(x.split("|")[0].replace("\n", "").strip(), x.split("|")[1].replace("\n", "").strip()) for x in
                 lines[1:]]
        # print(len(pairs), pairs)
    return pairs


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def clean_text(text):
    """
    This function takes as input a text on which several
    NLTK algorithms will be applied in order to preprocess it
    """
    tokens = word_tokenize(text)
    # Remove the punctuations
    tokens = [word for word in tokens if word.isalpha()]
    # Lower the tokens
    tokens = [word.lower() for word in tokens]
    # Remove stopword
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    # Lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    return [" ".join(tokens).strip()]


def get_weights(n):
    if n == 1:
        return (1.0,)
    elif n == 2:
        return (0.5, 0.5)
    elif n == 3:
        return (0.33, 0.33, 0.33)
    else:
        return (0.25, 0.25, 0.25, 0.25)


def get_bleu_score(hypothesis, reference):
    # hypothesis = [normalize_text(hypothesis)]
    # reference = [normalize_text(reference)]
    hypothesis = [hypothesis]
    reference = [reference]
    hypothesis = hypothesis[0].split(" ")
    reference = reference[0].split(" ")
    min_len = min(len(hypothesis), len(reference))
    weights = get_weights(min_len)
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    return bleu_score


def get_sacrebleu_score(hypothesis, reference):
    # hypothesis = [normalize_text(hypothesis)]
    # reference = [normalize_text(reference)]
    hypothesis = [hypothesis]
    reference = [reference]
    bleu = BLEU(lowercase=True, effective_order=True)
    sacrebleu_score = bleu.sentence_score(hypothesis[0], [reference[0]])
    # bleu = BLEU()
    # sacrebleu_score = bleu.corpus_score(hypothesis, [reference])

    return sacrebleu_score.score / 100


def get_bert_score(hypothesis, reference):
    global bert_scorer
    hypothesis = [hypothesis.lower()]
    reference = [reference.lower()]
    # hypothesis = clean_text(hypothesis)
    # reference = clean_text(reference)
    P, R, F1 = bert_scorer.score(hypothesis, reference)
    bert_score = float(F1.mean())

    return bert_score


def compute_f1(prediction, truth):
    # hypothesis = [normalize_text(hypothesis)]
    # reference = [normalize_text(reference)]
    # pred_tokens = normalize_text(prediction).split()
    # truth_tokens = normalize_text(truth).split()
    pred_tokens = prediction.lower().split()
    truth_tokens = truth.lower().split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return (2 * (prec * rec) / (prec + rec))


if __name__ == '__main__':
    data = {ABRAHAM_LINCOLN: readlines(PATH + 'abraham lincoln.txt'),
            ALBERT_EINSTEIN: readlines(PATH + 'albert einstein.txt'),
            ISAAC_ASIMOV: readlines(PATH + 'isaac asimov.txt'),
            NIKOLA_TESLA: readlines(PATH + 'nikola tesla.txt')}

    # hypothesis = ['Ana are mere']
    # reference = ['Ana are pere']
    # bleu_score = get_bleu_score(hypothesis, reference)
    # print(bleu_score)
    # sacrebleu_score = get_sacrebleu_score(hypothesis, reference)
    # print(sacrebleu_score)
    # bert_score = get_bert_score(hypothesis, reference)
    # print(bert_score)

    only_summary = True
    top_k = 5
    print("- only_summary =", only_summary)
    print("- dataset path =", PATH)
    print('- model_name = "deepset/deberta-v3-large-squad2"')
    if not only_summary:
        print('- top_k =', top_k)

    for agent in data:
        if only_summary:
            agent_name, sentences, page_content = read_data_from_wiki(agent, summary=True)
        else:
            agent_name, sentences, page_content = read_data_from_wiki(agent)
            model_name = "multi-qa-mpnet-base-cos-v1"
            bi_encoder = SentenceTransformer(model_name)
            paragraphs = get_paragraphs(page_content)
            paragraphs_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True,
                                                      show_progress_bar=True)
        # print("# of sentences=", len(sentences))
        lens = [len(sentence.split(" ")) for sentence in sentences]
        # print("# of words on average", sum(lens) / len(sentences))
        # print("# of characters=", len(page_content))

        pairs = data[agent]

        bleu_score_total = 0
        bert_score_total = 0
        f1_score_total = 0
        with open(PATH + agent_name + "_predictions.txt", 'w', encoding="utf-8") as f:
            f.write("Q, A, P\n")
            for question, answer in pairs:
                answer = answer[:-1]
                if only_summary:
                    predicted_answer = bert_answer(question, page_content)
                else:
                    context = get_context_based_on_question2(question, page_content, top_k=top_k, bi_encoder=bi_encoder,
                                                             paragraphs_embeddings=paragraphs_embeddings)
                    predicted_answer = bert_answer(question, context)
                bleu_score = get_sacrebleu_score(answer, predicted_answer)
                bert_score = get_bert_score(answer, predicted_answer)
                f1_score = compute_f1(predicted_answer, answer)
                """print("Question:", question)
                print("Answer:", answer)
                print("Predicted answer:", predicted_answer)
                print("BLEU Score:", bleu_score)
                print("BERT Score:", bert_score)
                print("F1 Score:", f1_score)"""
                bleu_score_total += bleu_score
                bert_score_total += bert_score
                f1_score_total += f1_score
                line = question + " | " + answer + " | " + predicted_answer + "\n"
                f.write(line)
        f.close()

        print(agent_name + " -> BLEU: " + "%.3f" % (bleu_score_total / len(pairs)))
        print(agent_name + " -> BERT: " + "%.3f" % (bert_score_total / len(pairs)))
        print(agent_name + " -> F1: " + "%.3f" % (f1_score_total / len(pairs)))
        print()

        # break
