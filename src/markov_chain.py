"""
Markovchain - next word preditction.

The core idea:
    1. Read training text and split into words
    2. For each word, trackw hat word comes after it
    3. Store counts: after "the", we see "cat" 5 times, "dog" 3 times
    4. Convert to probabilities: P(cat|the)= 5/8, P(dog|the)= 3/8
    5. Use these probabilities to predict

Most of this code is basic probability
ps- this is my first time using fp, so code might be inefficient
"""

from collections import defaultdict


class Markovchain:
    """
    This is a markov chain model for predicting the next word.

    Data structures used:
            - self.transistions: {word:{next_word:count,...}}
        Example: {"the"{"cat": 5, "dog":3, "mat":2}}
        After "the", we've seen cat 5 times and dog 3 times

    - self.vocabulary: this is a set containing all the unique words we have seen

    """

    def __init__(self) -> None:
        """Initialize an empty markov chain"""

        self.transitions = defaultdict(lambda: defaultdict(int))

        self.vocabulary = set()

        self.transistionCount = 0

    def tokenize(self, text):
        """
        Convert text string into a list of word (tokens)
        My approach is to just normalize everything(basically lowercase everything) and split on whitespace

        Args:
            text (str): Raw text to tokenize

        Returns:
            list: List of words

        Example:
            "The cat sat on the bed" -> ["the", "cat", "sat", "on", "the", "bed"]
        """

        text = text.lower()

        words = text.split()

        return words
