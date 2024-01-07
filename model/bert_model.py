from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import time
import torch


def main():
    """
    encoding = tokenizer.encode_plus(text, add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    """
    print(torch.__version__)
    print(torch.cuda.is_available())
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    start = time.time()

    question = "How many parameters does BERT-large have?"
    answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

    input_ids = tokenizer.encode(question, answer_text)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(len(tokens))
    print((tokens))

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # Run our example through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor(
                                         [segment_ids]), return_dict=False)  # The segment IDs to differentiate question from answer_text

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end + 1])

    print('Answer: "' + answer + '"')

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
