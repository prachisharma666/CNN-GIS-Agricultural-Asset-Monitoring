from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from .models import Task

@login_required
def task_list(request):
    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        priority = request.POST.get("priority", "medium")
        due_date = request.POST.get("due_date") or None
        if title:
            Task.objects.create(owner=request.user, title=title, priority=priority, due_date=due_date)
        return redirect("task_list")

    tasks = Task.objects.filter(owner=request.user).order_by("-created_at")
    return render(request, "tasks/task_list.html", {"tasks": tasks})

@login_required
def toggle_task(request, task_id):
    task = get_object_or_404(Task, id=task_id, owner=request.user)
    task.done = not task.done
    task.save()
    return redirect("task_list")

@login_required
def delete_task(request, task_id):
    task = get_object_or_404(Task, id=task_id, owner=request.user)
    task.delete()
    return redirect("task_list")