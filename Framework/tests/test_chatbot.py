import unittest
import json
from app import app

class TestChatbotAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_answer_question(self):
        """Test valid question returns answer with status 200."""
        response = self.app.post('/ask', json={
            'question': 'What are the symptoms of diabetes?'
        })
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn('answer', data)
        self.assertIn('score', data)

    def test_missing_question(self):
        """Test empty JSON returns 400 error."""
        response = self.app.post('/ask', json={})
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
