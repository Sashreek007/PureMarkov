#!/bin/bash

# Create directory
mkdir -p ~/MarkovData
cd ~/MarkovData

echo "Downloading books from Project Gutenberg..."
echo "This will take 5-15 minutes..."
echo ""

# Download top 100 books
for i in {1..100}; do
  wget -q https://www.gutenberg.org/cache/epub/$i/pg$i.txt
  echo "Downloaded book $i"
done

echo ""
echo "Combining all books into one file..."

# Combine all into one file
cat pg*.txt >gutenberg_combined.txt

echo ""
echo "Done! Dataset info:"
wc -w gutenberg_combined.txt

echo ""
echo "File location: ~/MarkovData/gutenberg_combined.txt"
