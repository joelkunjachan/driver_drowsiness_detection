from django.db import models

class registerr(models.Model):
    name=models.CharField(max_length=150)
    email=models.CharField(max_length=150)
    password=models.CharField(max_length=150)

class uploads(models.Model):
    u_id=models.CharField(max_length=150)
    file=models.FileField(max_length=200)
    result=models.CharField(max_length=150)

class feedbacks(models.Model):
    feedback=models.CharField(max_length=150)
    user=models.CharField(max_length=150)