# Generated by Django 3.2 on 2022-10-15 20:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0025_gdpforecast'),
    ]

    operations = [
        migrations.DeleteModel(
            name='AlphaVantageHistory',
        ),
    ]
