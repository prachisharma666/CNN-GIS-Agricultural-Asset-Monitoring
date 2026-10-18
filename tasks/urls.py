from django.urls import path

from . import views

urlpatterns = [
    path("", views.predict, name="predict"),
    path("claims/", views.claims, name="claims"),
    path("claims/<int:claim_id>/", views.claim_detail, name="claim_detail"),
    path("claims/<int:claim_id>/delete/", views.delete_claim, name="delete_claim"),
]