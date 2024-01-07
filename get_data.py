import spacy
import wikipedia
import re
from sentence_transformers import SentenceTransformer, util
import torch
import json
import random

nlp = spacy.load("en_core_web_sm")  # en_core_web_trf is slower but probably better (maybe with spacy-gpu is faster)


def read_data_from_file(filename):
    sentences = []
    with open(filename, "r", encoding="UTF-8") as input_file:
        doc = input_file.read()
        doc = doc.rstrip()
        parsed_doc = nlp(doc)

        for token in parsed_doc.sents:
            sentence = token.text.strip()

            # print(sentence)
            sentences.append(sentence)

    return sentences


def read_data_from_wiki(personality, summary=False):
    sentences = []
    if not summary:
        agent_name, content = get_wikipedia_article(personality)
    else:
        agent_name, content = get_wikipedia_article_summary(personality)

    doc = content.rstrip()
    parsed_doc = nlp(doc)

    for token in parsed_doc.sents:
        sentence = token.text.strip()

        # print(sentence)
        sentences.append(sentence)

    return agent_name, sentences, content


def get_paragraphs(text):
    text = text.replace('\n', '')
    paragraphs = re.findall('.*?(?<![A-Z])\.(?=[A-Z]|$)', text, re.DOTALL)
    return paragraphs


def get_paragraphs_history(text):
    text = text.split('\n')
    paragraphs = [t for t in text if t != '']
    return paragraphs


def get_context_based_on_question1(question, text, bi_encoder=None):
    model_name = 'nq-distilbert-base-v1'
    bi_encoder = SentenceTransformer(model_name)
    top_k = 5  # Number of passages we want to retrieve with the bi-encoder
    paragraphs = get_paragraphs(text)
    paragraphs_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)

    top_k = min(5, len(paragraphs))
    top_k = min(top_k, len(paragraphs))
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(question_embedding, paragraphs_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", question)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(paragraphs[idx], "(Score: {:.4f})".format(score))


def get_context_based_on_question2(question, text, top_k=5, bi_encoder=None, paragraphs_embeddings=None, is_history=False):
    # model_name = 'nq-distilbert-base-v1'
    if bi_encoder is None:
        model_name = "multi-qa-mpnet-base-cos-v1"
        bi_encoder = SentenceTransformer(model_name)
    top_k = top_k  # Number of passages we want to retrieve with the bi-encoder
    if not is_history:
        paragraphs = get_paragraphs(text)
    else:
        paragraphs = get_paragraphs_history(text)
    if paragraphs_embeddings is None:
        paragraphs_embeddings = bi_encoder.encode(paragraphs, convert_to_tensor=True, show_progress_bar=True)

    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    top_k = min(top_k, len(paragraphs))
    hits = util.semantic_search(question_embedding, paragraphs_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Output of top-k hits
    s = 0
    context = ""
    for hit in hits:
        # ("\t{:.3f}\t{}".format(hit['score'], paragraphs[hit['corpus_id']]))
        s += len(paragraphs[hit['corpus_id']].split(" "))
        context += paragraphs[hit['corpus_id']] + "\n"

    # print("# of words=", s)
    # print("\n\n========\n")

    return context


def get_wikipedia_article(personality, language="en"):
    wikipedia.set_lang(language)
    nt = wikipedia.page(personality, auto_suggest=False)

    page_title = nt.title
    page_content = nt.content
    page_content = remove_references(page_content)
    page_content = remove_paragraph_titles(page_content)

    # print()
    # print(nt.url)
    # print(page_title)
    # print("######################################################################################################\n")
    # print(page_content)

    return page_title, page_content


def get_wikipedia_article_summary(personality, language="en"):
    wikipedia.set_lang(language)
    nt = wikipedia.summary(personality, auto_suggest=False)

    page_content = remove_references(nt)
    page_content = remove_paragraph_titles(page_content)

    # print()
    # print(personality)
    # print("######################################################################################################\n")
    # print(page_content)

    return personality, page_content


def remove_references(page_content):
    page_content = re.sub(r'(=)*(?<=(=))( See also )(?=(=))(.\n*)*', "", page_content)
    page_content = page_content.strip()

    return page_content


def remove_paragraph_titles(page_content):
    page_content = re.sub(r'(?<=(=)).*(?=(=))', "", page_content)
    page_content = re.sub(r'(==)*', "", page_content)
    page_content = page_content.strip()

    return page_content


def remove_paragraph_titles2(page_content):
    page_content = re.sub(r'(?<=(=))[a-zA-Z0-9 ]*(?=(=))', "", page_content)
    page_content = re.sub(r'(==)*', "", page_content)
    page_content = page_content.strip()

    return page_content


def get_intents():
    data_file = "data/my_intents.json"
    data_file = open(data_file).read()
    intents = json.loads(data_file)
    patterns = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)

    return intents, patterns


def get_label_by_pattern(intents, pattern):
    for intent in intents['intents']:
        for pattern_el in intent['patterns']:
            if pattern[0] == pattern_el:
                return intent["label"]

    return "noanswer"


def get_response_by_question(intents, patterns, question, bi_encoder=None, patterns_embeddings=None):
    if bi_encoder is None:
        model_name = "multi-qa-mpnet-base-cos-v1"
        bi_encoder = SentenceTransformer(model_name)
    top_k = 1  # Number of passages we want to retrieve with the bi-encoder
    if patterns_embeddings is None:
        patterns_embeddings = bi_encoder.encode(patterns, convert_to_tensor=True, show_progress_bar=True)

    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    top_k = min(top_k, len(patterns))
    hits = util.semantic_search(question_embedding, patterns_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Output of top-k hits
    context = []
    for hit in hits:
        context.append(patterns[hit['corpus_id']])

    label = get_label_by_pattern(intents, context)
    # print("label:", label)
    response = None
    for intent in intents['intents']:
        if (intent['label'] == label):
            response = random.choice(intent['responses'])
            break
    return response


def is_relevant_question(question, text, top_k=1, bi_encoder=None, paragraphs_embeddings=None, is_history=False):
    # model_name = 'nq-distilbert-base-v1'
    if bi_encoder is None:
        model_name = "multi-qa-mpnet-base-cos-v1"
        bi_encoder = SentenceTransformer(model_name)
    top_k = top_k  # Number of passages we want to retrieve with the bi-encoder
    if paragraphs_embeddings is None:
        if not is_history:
            paragraphs = get_paragraphs(text)
        else:
            paragraphs = get_paragraphs_history(text)
        paragraphs_embeddings = bi_encoder.encode(paragraphs, show_progress_bar=True)

    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(question)
    # top_k = min(top_k, len(paragraphs))
    hits = util.semantic_search(question_embedding, paragraphs_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Output of top-k hits
    score = 0
    for hit in hits:
        # print(hit['score'])
        score = hit['score']
        break

    return score


if __name__ == '__main__':
    # get_wikipedia_article("Nikola Tesla")
    """agent_name, sentences, content = read_data_from_wiki("Nikola Tesla")
    # print(content)
    context = get_context_based_on_question2("Who was Nikola Tesla?", content)
    print(context)"""

    """intents, patterns = get_intents()
    question = "hello!"
    model_name = "multi-qa-mpnet-base-cos-v1"
    bi_encoder = SentenceTransformer(model_name)
    patterns_embeddings = bi_encoder.encode(patterns, convert_to_tensor=True, show_progress_bar=True)
    response = get_response_by_question(intents, patterns, question, bi_encoder=bi_encoder,
                                        patterns_embeddings=patterns_embeddings)
    print("response:", response)"""

    agent_name, sentences, content = read_data_from_wiki("Nikola Tesla")
    score = is_relevant_question("How are you?", content)
    print(score)
