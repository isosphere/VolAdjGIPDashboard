# Generated by Django 3.2.21 on 2025-05-08 01:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0028_signaltimeseries'),
    ]

    operations = [
        migrations.DeleteModel(
            name='BitfinexHistory',
        ),
        migrations.DeleteModel(
            name='SignalTimeSeries',
        ),
    ]
