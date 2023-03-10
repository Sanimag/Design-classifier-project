# Generated by Django 4.1.7 on 2023-03-02 08:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pr_title', models.CharField(max_length=500)),
                ('pr_description', models.TextField()),
                ('pr_image', models.ImageField(upload_to='product_images/')),
            ],
        ),
    ]
