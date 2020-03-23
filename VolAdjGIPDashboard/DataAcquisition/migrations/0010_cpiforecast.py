# Generated by Django 2.2.6 on 2020-03-23 03:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0009_auto_20191111_2021'),
    ]

    operations = [
        migrations.CreateModel(
            name='CPIForecast',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quarter_end_date', models.DateField()),
                ('date', models.DateField()),
                ('cpi', models.FloatField()),
            ],
        ),
    ]
