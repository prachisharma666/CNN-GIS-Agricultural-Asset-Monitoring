from django.contrib.auth import get_user_model
from django.test import TestCase


class HomeTests(TestCase):
    def setUp(self):
        get_user_model().objects.create_user("asha", password="pw-12345-xyz")

    def test_home_requires_login(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res["Location"].startswith("/accounts/login/"))

    def test_home_after_login(self):
        self.client.login(username="asha", password="pw-12345-xyz")
        res = self.client.get("/")
        self.assertContains(res, "Cropland classifier")
        self.assertContains(res, "asha")

    def test_login_page(self):
        self.assertContains(self.client.get("/accounts/login/"), "Log in")