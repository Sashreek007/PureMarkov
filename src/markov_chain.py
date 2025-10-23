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
import random


class Markovchain:
    """
    This is a markov chain model for predicting the next word.

    Data structures used:
            - self.transistions: {word:{next_word:count,...}}
        Example: {"the"{"cat": 5, "dog":3, "mat":2}}
        After "the", we've seen cat 5 times and dog 3 times

    - self.vocabulary: this is a set containing all the unique words we have seen

    """

    def __init__(self, order=1) -> None:
        """Initialize an empty markov chain
        Args:
        order: Length of context to consider (default 1)
               1 = unigram (single word context)
               2 = bigram (two word context, recommended)
               3 = trigram (three word context)
               Higher values give better accuracy but need more data

        """
        self.order = order

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

    def train(self, text):
        """
        This is not exactly machine learning way of training. Its more like a count-based statistical training

        Process:
            1. Tokenize text into words
            2. Create sliding windows of (order + 1) words
            3. Extract (context, next_word) pairs from windows
            4. Count occurrences of each transition
            5. Build vocabulary set

        Args:
            text: Training text to learn from

        Example for order=1:
            model = Markovchain(order=1)
             model.train("the cat sat on the bed")
             After training, transitions will be:
             {
                 "the": {"cat": 1, "bed": 1},
                 "cat": {"sat": 1},
                 "sat": {"on": 1},
                 "on": {"the": 1}
             }

        Example for order=2:
             model = Markovchain(order=2)
             model.train("the cat sat on the bed")
             After training, transitions will be:
             {
                 ("the", "cat"): {"sat": 1},
                 ("cat", "sat"): {"on": 1},
                 ("sat", "on"): {"the": 1},
                 ("on", "the"): {"bed": 1}
             }
        """

        # Tokenize
        words = self.tokenize((text))

        for i in range(len(words) - self.order):
            context = tuple(words[i : i + self.order])
            nextWord = words[i + self.order]
            self.transitions[context][nextWord] += 1
            # the number of times we saw a new word
            self.transistionCount += 1

            self.vocabulary.add(nextWord)
            for word in context:
                self.vocabulary.add(word)

        print(f"Trained on {len(words)} words")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Unique transitions learned: {len(self.transitions)}")
        print(f"Order: {self.order}")

    def getTransitions(self, context):
        """
        Get all possible next words after a given context with their counts.

        Args:
            context: The context to look up
                     - For order=1: a string, e.g., "cat"
                     - For order=2: a tuple of 2 words, e.g., ("the", "cat")
                     - For order=3: a tuple of 3 words, e.g., ("the", "cat", "sat")

        Returns:
            Dictionary where keys are possible next words and values are counts
            Returns empty dict if context not found in training data

        Example for order=1:
            model.getTransitions("the") ->{"cat": 2, "dog": 3}

        Example for order=2:
            model.getTransitions(("the", "cat"))->{"sat": 2, "and": 1}
        """
        if isinstance(context, str):
            context = (context,)

        return dict(self.transitions.get(context, {}))

    def predictNext(self, context, method="max"):
        """
        Predict the next word after a given context.

        Args:
            context: The context to predict from
                     - For order=1: a string, e.g., "cat"
                     - For order=2: a tuple of 2 words, e.g., ("the", "cat")
                     - For order=3: a tuple of 3 words, e.g., ("the", "cat", "sat")

            method:
                'max' - Returns the most likely next word (deterministic)
                'sample' - Returns a random word weighted by probability (probabilistic)

        Returns:
            Predicted next word as a string, or None if context not found in training data

        Example for order=1:
            model.predictNext("the", method="max")->'cat'

            model.predictNext("the", method="sample")->'dog'
        Example for order=2:
            model.predictNext(("the", "cat"), method="max")->'sat'

        """
        if isinstance(context, str):
            context = (context,)

        nextWord = self.transitions.get(context, {})
        if not nextWord:
            return None

        if method == "max":
            maxWord = max(nextWord, key=nextWord.get)
            return maxWord

        elif method == "sample":
            total = sum(nextWord.values())
            probabilites = {w: count / total for w, count in nextWord.items()}

            words = list(probabilites.keys())
            probs = list(probabilites.values())

            chose = random.choices(words, weights=probs, k=1)[0]
            return chose

    def generateText(self, startContext, length=10, method="max"):
        """
        Generate text starting from a given context.

        Args:
            startContext: Initial context to start generation
                         - For order=1: a string, e.g., "the"
                         - For order=2: a tuple of 2 words, e.g., ("the", "cat")

            length: Number of words to generate after the starting context
            method: 'max' for deterministic, 'sample' for probabilistic

        Returns:
            Generated text as a string

        Example for order=1:
            model.generateText("the", length=8, method="max")->'the cat sat on the mat the cat sat'

        Example for order=2:
            model.generateText(("the", "cat"), length=8, method="max")->'the cat sat on the mat and looked outside'
        """

        if isinstance(startContext, str):
            context = (startContext,)
        else:
            context = startContext

        generatedWords = list(context)

        for _ in range(length):
            currentContext = tuple(generatedWords[-self.order :])
            nextWord = self.predictNext(currentContext, method=method)

            if nextWord is None:
                break

            generatedWords.append(nextWord)
        return " ".join(generatedWords)

    def getStats(self):
        """
        Get model statistics

        Returns:
            Dictionary containing model information

        Example:
            model.getStats()
            {
                'order': 2,
                'vocabulary_size': 42,
                'unique_contexts': 38,
                'total_transitions': 100
            }
        """
        return {
            "order": self.order,
            "vocabularySize": len(self.vocabulary),
            "uniqueContexts": len(self.transitions),
            "totalTransitions": self.transistionCount,
        }
