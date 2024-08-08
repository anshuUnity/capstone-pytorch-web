from django.db import models

# Create your models here.
class Prediction(models.Model):
    image = models.ImageField(upload_to="images")
    predction = models.CharField(max_length=55, blank=True, null=True)
