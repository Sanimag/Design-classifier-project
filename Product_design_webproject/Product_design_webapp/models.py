from django.db import models

# Create your models here.
class Product(models.Model):
    pr_title = models.CharField(max_length = 500)
    pr_description = models.TextField()
    pr_image = models.ImageField(upload_to='product_images/')
