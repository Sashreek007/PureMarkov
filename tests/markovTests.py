"""
Unit tests for Markovchain - next word prediction.

Tests cover:
    - Tokenization
    - Training and vocabulary building
    - Transition counting
    - Prediction (max and sample methods)
    - Text generation
    - Model statistics
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.markov_chain import Markovchain


class TestTokenization(unittest.TestCase):
    """Test text tokenization."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)

    def testTokenizeBasic(self):
        """Test basic tokenization."""
        text = "The cat sat on the mat"
        tokens = self.model.tokenize(text)
        expected = ["the", "cat", "sat", "on", "the", "mat"]
        self.assertEqual(tokens, expected)

    def testTokenizeLowercase(self):
        """Test that tokenization converts to lowercase."""
        text = "THE CAT SAT ON THE MAT"
        tokens = self.model.tokenize(text)

        for token in tokens:
            self.assertTrue(token.islower())

    def testTokenizeWithPunctuation(self):
        """Test tokenization with punctuation."""
        text = "the cat sat, on the mat."
        tokens = self.model.tokenize(text)
        # Should split on whitespace but keep punctuation attached
        self.assertIn("sat,", tokens)
        self.assertIn("mat.", tokens)


class TestTraining(unittest.TestCase):
    """Test model training."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)
        self.simpleText = "the cat sat on the mat the dog sat on the floor"

    def testTrainingBuildsVocabulary(self):
        """Test that training builds vocabulary."""
        self.model.train(self.simpleText)
        self.assertGreater(len(self.model.vocabulary), 0)
        self.assertIn("the", self.model.vocabulary)
        self.assertIn("cat", self.model.vocabulary)

    def testTrainingCountsTransitions(self):
        """Test that transitions are counted."""
        self.model.train(self.simpleText)

        transitions = self.model.getTransitions("the")
        self.assertGreater(len(transitions), 0)

    def testTrainingOrder1(self):
        """Test training with order=1."""
        model = Markovchain(order=1)
        model.train("a b c a b c")

        # Should have transitions from single words
        transitions = model.getTransitions("a")
        self.assertEqual(transitions["b"], 2)

    def testTrainingOrder2(self):
        """Test training with order=2."""
        model = Markovchain(order=2)
        model.train("a b c a b c")

        # Should have transitions from word pairs
        transitions = model.getTransitions(("a", "b"))
        self.assertEqual(transitions["c"], 2)

    def testTrainingOrder3(self):
        """Test training with order=3."""
        model = Markovchain(order=3)
        model.train("a b c d a b c d")

        # Should have transitions from word triplets
        transitions = model.getTransitions(("a", "b", "c"))
        self.assertEqual(transitions["d"], 2)


class TestTransitions(unittest.TestCase):
    """Test transition lookup."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)
        self.text = "the cat sat on the mat the dog sat on the floor"

    def testGetTransitionsReturnsDict(self):
        """Test that getTransitions returns a dictionary."""
        self.model.train(self.text)

        transitions = self.model.getTransitions("the")
        self.assertIsInstance(transitions, dict)

    def testGetTransitionsUnknownWord(self):
        """Test getTransitions with unknown word returns empty dict."""
        self.model.train(self.text)

        transitions = self.model.getTransitions("unknown")
        self.assertEqual(transitions, {})

    def testGetTransitionsStringContext(self):
        """Test getTransitions with string context (order=1)."""
        self.model.train(self.text)

        transitions = self.model.getTransitions("the")
        self.assertGreater(len(transitions), 0)

    def testGetTransitionsTupleContext(self):
        """Test getTransitions with tuple context (order=2)."""
        model = Markovchain(order=2)
        model.train(self.text)

        transitions = model.getTransitions(("the", "cat"))
        self.assertIsInstance(transitions, dict)


class TestPrediction(unittest.TestCase):
    """Test next word prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)
        self.text = "the cat sat on the mat the dog sat on the floor"

    def testPredictNextReturnsString(self):
        """Test that predictNext returns a string."""
        self.model.train(self.text)

        pred = self.model.predictNext("the", method="max")
        self.assertIsInstance(pred, str)

    def testPredictNextUnknownWord(self):
        """Test predictNext with unknown word returns None."""
        self.model.train(self.text)

        pred = self.model.predictNext("xyz", method="max")
        self.assertIsNone(pred)

    def testPredictNextMaxDeterministic(self):
        """Test that max method is deterministic."""
        self.model.train(self.text)

        pred1 = self.model.predictNext("the", method="max")
        pred2 = self.model.predictNext("the", method="max")

        self.assertEqual(pred1, pred2)

    def testPredictNextSampleIsValid(self):
        """Test that sample method returns valid word."""
        self.model.train(self.text)

        for _ in range(5):
            pred = self.model.predictNext("the", method="sample")
            self.assertIn(pred, self.model.getTransitions("the").keys())

    def testPredictNextOrder2(self):
        """Test prediction with order=2."""
        model = Markovchain(order=2)
        model.train(self.text)

        pred = model.predictNext(("the", "cat"), method="max")
        self.assertIsNotNone(pred)

    def testPredictNextOrder3(self):
        """Test prediction with order=3."""
        model = Markovchain(order=3)
        model.train(self.text)

        pred = model.predictNext(("the", "cat", "sat"), method="max")
        # Might be None if context not found, but should not error
        self.assertTrue(pred is None or isinstance(pred, str))


class TestTextGeneration(unittest.TestCase):
    """Test text generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)
        self.text = """
        the cat sat on the mat
        the dog sat on the floor
        the bird sat on the tree
        """

    def testGenerateTextReturnsString(self):
        """Test that generateText returns a string."""
        self.model.train(self.text)

        generated = self.model.generateText("the", length=5, method="max")
        self.assertIsInstance(generated, str)

    def testGenerateTextStartsWithContext(self):
        """Test that generated text starts with start context."""
        self.model.train(self.text)

        generated = self.model.generateText("cat", length=5, method="max")
        words = generated.split()
        self.assertEqual(words[0], "cat")

    def testGenerateTextDeterministic(self):
        """Test that deterministic generation is consistent."""
        self.model.train(self.text)

        gen1 = self.model.generateText("the", length=10, method="max")
        gen2 = self.model.generateText("the", length=10, method="max")

        self.assertEqual(gen1, gen2)

    def testGenerateTextLength(self):
        """Test generated text has correct length."""
        self.model.train(self.text)

        for length in [5, 10, 15]:
            generated = self.model.generateText("the", length=length, method="max")
            wordCount = len(generated.split())

            # Should be at least 1, at most length
            self.assertGreater(wordCount, 0)
            self.assertLessEqual(wordCount, length + 1)

    def testGenerateTextOrder2(self):
        """Test text generation with order=2."""
        model = Markovchain(order=2)
        model.train(self.text)

        generated = model.generateText(("the", "cat"), length=5, method="max")
        self.assertGreater(len(generated), 0)

    def testGenerateTextSampleMethod(self):
        """Test text generation with sample method."""
        self.model.train(self.text)

        for _ in range(5):
            generated = self.model.generateText("the", length=8, method="sample")
            self.assertGreater(len(generated), 0)
            self.assertTrue(generated.startswith("the"))


class TestStatistics(unittest.TestCase):
    """Test model statistics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = Markovchain(order=1)
        self.text = "the cat sat on the mat the dog sat on the floor"

    def testGetStatsReturnsDict(self):
        """Test that getStats returns a dictionary."""
        self.model.train(self.text)

        stats = self.model.getStats()
        self.assertIsInstance(stats, dict)

    def testGetStatsHasRequiredKeys(self):
        """Test that stats dictionary has required keys."""
        self.model.train(self.text)

        stats = self.model.getStats()

        self.assertIn("order", stats)
        self.assertIn("vocabularySize", stats)
        self.assertIn("uniqueContexts", stats)
        self.assertIn("totalTransitions", stats)

    def testGetStatsValuesArePositive(self):
        """Test that stats values are positive."""
        self.model.train(self.text)

        stats = self.model.getStats()

        self.assertGreater(stats["order"], 0)
        self.assertGreater(stats["vocabularySize"], 0)
        self.assertGreater(stats["uniqueContexts"], 0)
        self.assertGreater(stats["totalTransitions"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def testFullWorkflowOrder1(self):
        """Test complete workflow with order=1."""
        model = Markovchain(order=1)
        text = "the cat sat on the mat the dog sat on the floor"

        # Train
        model.train(text)

        # Get stats
        stats = model.getStats()
        self.assertGreater(stats["vocabularySize"], 0)

        # Make prediction
        pred = model.predictNext("the", method="max")
        self.assertIsNotNone(pred)

        # Generate text
        generated = model.generateText("the", length=8, method="max")
        self.assertGreater(len(generated), 0)

    def testFullWorkflowOrder2(self):
        """Test complete workflow with order=2."""
        model = Markovchain(order=2)
        text = "the cat sat on the mat the dog sat on the floor"

        # Train
        model.train(text)

        # Make prediction
        pred = model.predictNext(("the", "cat"), method="max")
        self.assertIsNotNone(pred)

        # Generate text
        generated = model.generateText(("the", "cat"), length=8, method="max")
        self.assertGreater(len(generated), 0)

    def testConsistencyAcrossRuns(self):
        """Test that model is consistent across multiple runs."""
        text = "the cat sat on the mat the dog sat on the floor"

        # Train two models
        model1 = Markovchain(order=1)
        model1.train(text)

        model2 = Markovchain(order=1)
        model2.train(text)

        # Should give same predictions
        pred1 = model1.predictNext("the", method="max")
        pred2 = model2.predictNext("the", method="max")

        self.assertEqual(pred1, pred2)


def runTests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestTextGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Markovchain Unit Tests")
    print("=" * 70 + "\n")

    result = runTests()

    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70 + "\n")

    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed!")
