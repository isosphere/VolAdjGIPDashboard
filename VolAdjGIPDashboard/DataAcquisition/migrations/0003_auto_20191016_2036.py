# Generated by Django 2.2.6 on 2019-10-17 00:36

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0002_auto_20191016_2010'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='quarterreturn',
            name='updated',
        ),
        migrations.AddField(
            model_name='quarterreturn',
            name='prices_updated',
            field=models.DateTimeField(default=datetime.datetime(2019, 10, 16, 20, 36, 21, 13500)),
            preserve_default=False,
        ),
    ]
