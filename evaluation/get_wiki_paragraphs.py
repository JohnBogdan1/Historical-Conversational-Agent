import spacy
import wikipedia
import re

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


def get_context_based_on_question(question, text):
    pass


def get_wikipedia_article(personality, language="en"):
    wikipedia.set_lang(language)
    nt = wikipedia.page(personality, auto_suggest=False)

    page_title = nt.title
    page_content = nt.content
    page_content = remove_references(page_content)
    page_content = remove_paragraph_titles(page_content)

    print()
    print(nt.url)
    print(page_title)
    print("######################################################################################################\n")
    # print(page_content)

    return page_title, page_content


def get_wikipedia_article_summary(personality, language="en"):
    wikipedia.set_lang(language)
    nt = wikipedia.summary(personality, auto_suggest=False)

    page_content = remove_references(nt)
    page_content = remove_paragraph_titles(page_content)

    print()
    print(personality)
    print("######################################################################################################\n")
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


def get_paragraphs(text):
    paragraphs = re.findall('.*?(?<![A-Z])\.(?=[A-Z]|$)', text, re.DOTALL)
    return paragraphs


if __name__ == '__main__':
    # get_wikipedia_article("Nikola Tesla")
    ABRAHAM_LINCOLN = 'abraham lincoln'
    ALBERT_EINSTEIN = 'albert einstein'
    ISAAC_ASIMOV = 'isaac asimov'
    NIKOLA_TESLA = 'nikola tesla'
    agents = [ABRAHAM_LINCOLN, ALBERT_EINSTEIN, ISAAC_ASIMOV, NIKOLA_TESLA]
    avg_total = 0
    for agent in agents:
        agent_name, sentences, content = read_data_from_wiki(agent)
        # print(content)

        paragraphs = get_paragraphs(content.replace('\n', ''))

        nr_p = len(paragraphs)
        nr_w_t = 0
        for paragraph in paragraphs:
            nr_w = len(paragraph.split(" "))
            nr_w_t += nr_w

        avg = nr_w_t / nr_p
        print("Average # words / paragraph:", avg)
        avg_total += avg

    print("\n[Total] Average # words / paragraph:", avg_total / len(agents))

    """for p in paragraphs:
        print(p)
        print()"""
