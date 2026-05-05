from celery import Celery

def make_celery():
    return Celery(
        "tasks",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0"
    )

celery = make_celery()

@celery.task
def hello_task():
    return "Celery is working!"