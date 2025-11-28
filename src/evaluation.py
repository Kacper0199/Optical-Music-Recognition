import logging
import Levenshtein

logger = logging.getLogger(__name__)


def evaluate_prediction(ground_truth, prediction):
    gt_tokens = _tokenize_abc(ground_truth)
    pred_tokens = _tokenize_abc(prediction)

    unique_tokens = list(set(gt_tokens + pred_tokens))
    token_map = {token: chr(i + 5000) for i, token in enumerate(unique_tokens)}

    gt_str = "".join([token_map[t] for t in gt_tokens])
    pred_str = "".join([token_map[t] for t in pred_tokens])

    dist = Levenshtein.distance(gt_str, pred_str)

    len_gt = len(gt_tokens)
    len_pred = len(pred_tokens)
    max_len = max(len_gt, len_pred)

    accuracy = 0.0
    correct_symbols = 0

    if max_len > 0:
        accuracy = 1.0 - (dist / max_len)
        correct_symbols = max(0, max_len - dist)

    return accuracy, correct_symbols, max_len, dist


def _tokenize_abc(text):
    if not text:
        return []

    tokens = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if len(line) > 2 and line[1] == ':' and line[0] in ['M', 'L', 'K', 'T', 'X']:
            parts = line.split(':', 1)
            key = parts[0].strip()
            val = parts[1].strip()
            tokens.append(f"{key}:{val}")
        else:
            line = line.replace('|', ' | ')
            line = line.replace('[', ' [ ')
            line = line.replace(']', ' ] ')

            parts = line.split()
            tokens.extend(parts)

    return tokens
