import unittest
from flask_app.app import create_app

# Initialize Flask app for testing
test_app = create_app()

test_app.testing = True

class FlaskAppTests(unittest.TestCase):
    """
    Functional tests for the Flask sentiment analysis application.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create a test client for the Flask application once for all tests.
        """
        cls.client = test_app.test_client()

    def test_home_page(self):
        """
        Verify that the home ('/') endpoint returns HTTP 200
        and contains the expected HTML title.
        """
        response = self.client.get('/')
        # Expect OK status and correct title in HTML
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        """
        Verify that submitting review text returns HTTP 200
        and that the response contains either 'Positive' or 'Negative'.
        """
        payload = dict(text="I love this!")
        response = self.client.post('/predict', data=payload)
        # Expect OK status
        self.assertEqual(response.status_code, 200)
        # The result should indicate sentiment
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,
            "Response should contain either 'Positive' or 'Negative'"
        )


if __name__ == '__main__':
    unittest.main()
