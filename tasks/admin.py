from django.contrib import admin
from .models import Task

@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ("title", "owner", "priority", "due_date", "done", "created_at")
    list_filter = ("done", "priority")