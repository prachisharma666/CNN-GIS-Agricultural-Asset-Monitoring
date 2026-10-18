from django.contrib import admin

from .models import Claim, Task


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ("title", "owner", "priority", "due_date", "done", "created_at")
    list_filter = ("done", "priority")


@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ("farm_id", "farmer_name", "crop_type", "claimed_area_ha", "priority", "owner", "created_at")
    list_filter = ("priority", "crop_type")
    search_fields = ("farm_id", "farmer_name")