# Generated by Django 3.2.21 on 2025-05-17 00:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0029_auto_20250507_2106'),
    ]

    operations = [
        migrations.AlterField(
            model_name='yahoohistory',
            name='ticker',
            field=models.CharField(max_length=36),
        ),
    ]
