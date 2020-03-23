# Generated by Django 2.2.6 on 2020-03-23 04:44

import datetime
from django.db import migrations, models


def forwards_func(apps, schema_editor):
    QuadReturn = apps.get_model("DataAcquisition", "QuadReturn")
    QuadReturn.objects.all().delete()

class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0010_cpiforecast'),
    ]

    operations = [
        migrations.AddField(
            model_name='quadreturn',
            name='data_start_date',
            field=models.DateField(default=datetime.date(1970, 1, 1)),
            preserve_default=False,
        ),
        migrations.AlterUniqueTogether(
            name='quadreturn',
            unique_together={('quarter_end_date', 'data_start_date', 'data_end_date', 'label')},
        ),
        migrations.RunPython(forwards_func)
    ]
