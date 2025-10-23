"""
Markovchain - next word preditction.

The core idea:
    1. Read training text and split into words
    2. For each word, trackw hat word comes after it
    3. Store counts: after "the", we see "cat" 5 times, "dog" 3 times
    4. Convert to probabilities: P(cat|the)= 5/8, P(dog|the)= 3/8
    5. Use these probabilities to predict
"""
