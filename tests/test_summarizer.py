import unittest

from summarizer import count_words, split_sentences, summarize_text, evaluate_against_reference


class SummarizerTests(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("First sentence. Second sentence! Third sentence?")), 3)

    def test_empty_input(self):
        self.assertEqual(summarize_text("   ")[0], "")

    def test_summary_sentences_come_from_source(self):
        text = ("Cats are common household pets. They often sleep for many hours. "
                "Some cats enjoy playing with toys. Cats need food, water, and care.")
        summary, ranking = summarize_text(text, "Short")
        self.assertTrue(summary)
        self.assertGreaterEqual(len(ranking), 1)
        for sentence in split_sentences(summary):
            self.assertIn(sentence, split_sentences(text))

    def test_word_count(self):
        self.assertEqual(count_words("NLP makes text useful."), 4)

    def test_rouge_perfect_match(self):
        text = "machine learning improves prediction"
        scores = evaluate_against_reference(text, text)
        self.assertAlmostEqual(scores["rouge1_f1"], 1.0)
        self.assertAlmostEqual(scores["rouge2_f1"], 1.0)
        self.assertAlmostEqual(scores["rougeL_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
