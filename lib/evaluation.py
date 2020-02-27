from itertools import zip_longest
import json


def evaluate(text_file, correct_file, submission_file, count=None):
    with open(text_file) as f:
        text = json.load(f)
    with open(correct_file) as f:
        correct = json.load(f)
    with open(submission_file) as f:
        submission = json.load(f)

    if count:
        text = text[:count]
        correct = correct[:count]
        submission = submission[:count]

    data = []
    for sent, cor, sub in zip_longest(text, correct, submission):
        for w, c, s in zip_longest(sent, cor, sub):
            if w in ['a', 'an', 'the']:
                if s is None or s[0] == w:
                    s = ['', float('-inf')]
                data.append((-s[1], s[0] == c, c is not None))
    data.sort()
    fp2 = 0
    fp = 0
    tp = 0
    all_mistakes = sum(x[2] for x in data)
    score = 0
    acc = 0
    for _, c, r in data:
        fp2 += not c
        fp += not r
        tp += c
        acc = max(acc, 1 - (0. + fp + all_mistakes - tp) / len(data))
        if fp2 * 1. / len(data) <= 0.02:
            score = tp * 1. / all_mistakes
    print ('target score = %.2f %%' % (score * 100))
    print ('accuracy (just for info) = %.2f %%' % (acc * 100))


# decode [1, 0.5], [2, 0.956554] ... to input for evaluation
def decode_evaluation(sentences, scores):
    result = []

    correction_targets = ['a', 'an', 'the']
    targets = [6, 39, 2]

    correction_index = 0

    for sentence in sentences:
        sentence_result = []
        for word in sentence:
            if word in targets:
                label_index = targets.index(word)
                current_correction_index = int(scores[correction_index][0])
                if label_index != current_correction_index:
                    sentence_result.append([correction_targets[current_correction_index], scores[correction_index][1]])
                else:
                    sentence_result.append(None)
                correction_index += 1
            else:
                sentence_result.append(None)
        result.append(sentence_result)

    return result
