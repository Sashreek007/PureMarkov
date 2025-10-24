#!/usr/bin/env python3
import sys
import time
import os

sys.path.insert(0, "./src")
from markov_chain import Markovchain

print("\n" + "=" * 60)
print("MARKOV CHAIN CLI")
print("=" * 60 + "\n")

# Ask for order
print("Select order:")
print("1. Order 1 (single word context)")
print("2. Order 2 (two word context)")
print("3. Order 3 (three word context)")

orderChoice = input("\nEnter choice (1-3): ").strip()
orderMap = {"1": 1, "2": 2, "3": 3}
if orderChoice not in orderMap:
    print("Invalid choice!")
    sys.exit()

order = orderMap[orderChoice]

# Auto-detect data file
dataPath = os.path.expanduser("./MarkovData/gutenberg_combined.txt")

if not os.path.exists(dataPath):
    filepath = input("\nEnter path to training file: ").strip()
else:
    filepath = dataPath
    print(f"\nUsing data file: {filepath}")

try:
    print("Reading file...")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"File loaded: {len(text.split())} words")

    # Train
    print(f"\nTraining model (order={order})...")
    startTime = time.time()

    model = Markovchain(order=order)
    model.train(text)

    trainTime = time.time() - startTime
    print(f"Training completed in {trainTime:.2f} seconds\n")

    # Show stats
    stats = model.getStats()
    print("=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    print(f"Order: {stats['order']}")
    print(f"Vocabulary size: {stats['vocabularySize']}")
    print(f"Unique contexts: {stats['uniqueContexts']}")
    print(f"Total transitions: {stats['totalTransitions']}")
    print("=" * 60 + "\n")

    # Interactive mode
    while True:
        print("\nOptions:")
        print("1. Predict next word")
        print("2. Generate text")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            if order == 1:
                word = input("Enter a word: ").strip().lower()
                pred = model.predictNext(word, method="max")
                if pred:
                    print(f"Next word: {pred}")
                else:
                    print("Word not found")
            else:
                contextStr = input(f"Enter {order} words: ").strip().lower()
                contextWords = contextStr.split()
                if len(contextWords) != order:
                    print(f"Enter exactly {order} words")
                    continue
                context = tuple(contextWords)
                pred = model.predictNext(context, method="max")
                if pred:
                    print(f"Next word: {pred}")
                else:
                    print("Context not found")

        elif choice == "2":
            length = input("Words to generate? ").strip()
            try:
                length = int(length)
            except:
                print("Invalid number")
                continue

            if order == 1:
                startWord = input("Start word: ").strip().lower()
                text = model.generateText(startWord, length=length, method="max")
            else:
                contextStr = input(f"Start with {order} words: ").strip().lower()
                contextWords = contextStr.split()
                if len(contextWords) != order:
                    print(f"Enter exactly {order} words")
                    continue
                context = tuple(contextWords)
                text = model.generateText(context, length=length, method="max")

            print(f"\nGenerated:\n{text}\n")

        elif choice == "3":
            print("Done!")
            break
        else:
            print("Invalid choice!")

except FileNotFoundError:
    print(f"Error: File not found")
except Exception as e:
    print(f"Error: {str(e)}")
