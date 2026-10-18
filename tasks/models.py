from django.db import models
from django.contrib.auth.models import User
ACRE_IN_HA = 0.40468564224
class Task(models.Model):
    PRIORITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
    ]

    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    priority = models.CharField(max_length=6, choices=PRIORITY_CHOICES, default="medium")
    due_date = models.DateField(null=True, blank=True)
    done = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
class Claim(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    farm_id = models.CharField(max_length=40, blank=True)
    farmer_name = models.CharField(max_length=120, blank=True)
    crop_type = models.CharField(max_length=60, blank=True)
    claimed_area_ha = models.FloatField()
    entered_value = models.FloatField()
    entered_unit = models.CharField(max_length=10, default="acre")
    geometry = models.JSONField()
    result = models.JSONField()
    priority = models.CharField(max_length=6, default="none")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.farm_id or f"Claim {self.pk}"

    @property
    def claimed_acres(self):
        return self.claimed_area_ha / ACRE_IN_HA

    @property
    def detected_acres(self):
        v = self.result.get("cropland_area_ha")
        return None if v is None else v / ACRE_IN_HA