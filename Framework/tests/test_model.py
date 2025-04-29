import unittest
from src.model.inference import predict_answer

class TestModelPrediction(unittest.TestCase):
    def test_predict_answer(self):
        """
        Test that predict_answer returns a string answer and a positive confidence score.
        """
        question = "What are the symptoms of diabetes?"
        answer, score = predict_answer(question)

        self.assertIsInstance(answer, str)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

if __name__ == '__main__':
    unittest.main()
