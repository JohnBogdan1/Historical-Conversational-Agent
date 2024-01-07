from bert_score import BERTScorer

bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def get_bert_score(hypothesis, reference):
    global bert_scorer
    hypothesis = [hypothesis.lower()]
    reference = [reference.lower()]
    # hypothesis = clean_text(hypothesis)
    # reference = clean_text(reference)
    P, R, F1 = bert_scorer.score(hypothesis, reference)
    bert_score = float(F1.mean())

    return bert_score


h = "After he moved to Switzerland in 1895, the following year"
r = "in 1895"

print(get_bert_score(h, r))
