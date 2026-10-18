import io

import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image

from . import ml
from .models import Claim


def png(size=(64, 64), seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype="uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class PredictTests(TestCase):
    def setUp(self):
        get_user_model().objects.create_user("u", password="pw-12345-xyz")

    def test_login_required(self):
        self.assertEqual(self.client.get("/").status_code, 302)

    def test_predict_flow(self):
        self.client.login(username="u", password="pw-12345-xyz")
        self.assertEqual(self.client.get("/").status_code, 200)
        res = self.client.post("/", {"chip": SimpleUploadedFile("c.png", png(), content_type="image/png")})
        r = res.context["result"]
        self.assertIn(r["label"], ml.CLASSES)
        self.assertAlmostEqual(sum(r["probabilities"]), 1.0, places=4)
        self.assertContains(res, "Cropland probability")
        self.assertContains(res, 'class="bar-fill" style="width: ')
        self.assertContains(res, 'data-label="')

    def test_any_image_size(self):
        self.client.login(username="u", password="pw-12345-xyz")
        for size in ((10, 10), (32, 32), (300, 200)):
            res = self.client.post("/", {"chip": SimpleUploadedFile("c.png", png(size), content_type="image/png")})
            self.assertIsNotNone(res.context["result"], size)

    def test_bad_uploads(self):
        self.client.login(username="u", password="pw-12345-xyz")
        self.assertContains(self.client.post("/", {"chip": SimpleUploadedFile("x.png", b"not an image")}), "not a readable image")
        self.assertContains(self.client.post("/", {}), "Choose an image first")

    def test_model_is_loaded_once(self):
        self.client.login(username="u", password="pw-12345-xyz")
        ml.get_model.cache_clear()
        for i in range(3):
            self.client.post("/", {"chip": SimpleUploadedFile("c.png", png(seed=i), content_type="image/png")})
        info = ml.get_model.cache_info()
        self.assertEqual((info.misses, info.hits), (1, 2))

    def test_prediction_matches_direct_keras_call(self):
        raw = png((32, 32), seed=5)
        direct = ml.get_model().predict(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype="float32")[None], verbose=0)[0]
        self.assertAlmostEqual(ml.predict_image(io.BytesIO(raw))["cropland_probability"], float(direct[1]), places=5)


class ClaimDetailTemplateTests(TestCase):
    def test_claim_detail_renders(self):
        user = get_user_model().objects.create_user("reviewer")
        claim = Claim.objects.create(
            owner=user,
            claimed_area_ha=1.0,
            entered_value=1.0,
            geometry={
                "type": "Polygon",
                "coordinates": [[[74.0, 20.15], [74.01, 20.15], [74.01, 20.16], [74.0, 20.15]]],
            },
            result={
                "date1": "2026-01-03",
                "date2": "2026-04-13",
                "area_consistency_pct": 95.0,
                "ndvi_d1": None,
                "ndvi_d2": None,
                "ndvi_change": None,
                "valid_cov_d1": 100.0,
                "valid_cov_d2": 100.0,
                "dist_road_km": None,
                "dist_river_km": None,
                "flags": [],
                "warnings": [],
            },
        )
        self.client.force_login(user)

        response = self.client.get(f"/claims/{claim.pk}/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Claim vs satellite evidence")
        self.assertContains(response, 'id="claim-geom"')


class TemplateTests(TestCase):
    def setUp(self):
        get_user_model().objects.create_user("asha", password="pw-12345-xyz")

    def test_login_page(self):
        res = self.client.get("/accounts/login/")
        self.assertContains(res, "csrfmiddlewaretoken")
        self.assertContains(res, "Log in to the cropland classifier")
        self.assertContains(res, "tasks/style.css")

    def test_main_page_structure(self):
        self.client.login(username="asha", password="pw-12345-xyz")
        res = self.client.get("/")
        for needle in ("asha", 'id="theme-btn"', 'action="/accounts/logout/"', 'enctype="multipart/form-data"', 'name="chip"', "No image classified yet"):
            self.assertContains(res, needle)

    def test_error_message_is_escaped_and_shown(self):
        self.client.login(username="asha", password="pw-12345-xyz")
        res = self.client.post("/", {"chip": SimpleUploadedFile("x.png", b"nope")})
        self.assertContains(res, "not a readable image")
        self.assertNotContains(res, "No image classified yet")

    def test_percent_has_dot_decimal_in_css_width(self):
        self.client.login(username="asha", password="pw-12345-xyz")
        res = self.client.post("/", {"chip": SimpleUploadedFile("c.png", png(), content_type="image/png")})
        self.assertRegex(res.content.decode(), r'style="width: \d+\.\d%"')