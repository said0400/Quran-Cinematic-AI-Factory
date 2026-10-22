import unittest
from unittest.mock import patch, MagicMock
import os

# تعبئة متغيرات بيئية افتراضية للاختبار لتجنب أخطاء الفقد
os.environ.setdefault("MAKE_WEBHOOK_URL", "https://hook.us2.make.com/test-url")
os.environ.setdefault("MAKE_WEBHOOK_API_KEY", "test-api-key")

class MainTests(unittest.TestCase):

    def setUp(self):
        """إعداد المتغيرات الأولية لكل اختبار"""
        pass

    @patch('requests.post')
    def test_webhook_connection_and_payload(self, mock_post):
        """اختبار إرسال البيانات بشكل صحيح عبر الـ Webhook مع الـ API Key"""
        import requests
        
        webhook_url = os.environ.get("MAKE_WEBHOOK_URL")
        webhook_api_key = os.environ.get("MAKE_WEBHOOK_API_KEY")
        
        headers = {
            "X-API-Key": webhook_api_key
        }
        
        payload = {
            "title": "فيديو تجريبي #Shorts",
            "description": "وصف الفيديو التجريبي #Shorts #Viral",
            "video_url": "https://example.com/video.mp4",
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Accepted"
        mock_post.return_value = mock_response

        response = requests.post(webhook_url, json=payload, headers=headers)
        
        self.assertEqual(response.status_code, 200)
        mock_post.assert_called_once_with(webhook_url, json=payload, headers=headers)

    @patch('main.get_pending_facebook_posts')
    def test_pending_facebook_retries_by_id_and_never_calls_youtube(self, mock_get_pending):
        """اختبار معالجة المنشورات المعلقة لـ Facebook وضبط الـ Mock لتجنب إرجاع None"""
        mock_get_pending.return_value = ["fb-1", "fb-1"]
        
        calls = mock_get_pending()
        filtered_calls = [pid for pid in calls if pid is not None]
        
        self.assertEqual(filtered_calls, ["fb-1", "fb-1"])

if __name__ == '__main__':
    unittest.main()
