# Generated by Django 2.2.6 on 2020-09-03 17:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0017_yahoohistory_realized_volatility_week'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='yahoohistory',
            name='realized_volatility_week',
        ),
    ]