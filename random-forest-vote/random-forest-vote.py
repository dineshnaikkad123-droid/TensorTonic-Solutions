from collections import Counter

def random_forest_vote(predictions):
    result = []

    for sample_votes in zip(*predictions):
        counts = Counter(sample_votes)
        max_count = max(counts.values())
        winners = [cls for cls, cnt in counts.items() if cnt == max_count]
        result.append(min(winners))

    return result