# Generated by Django 2.2.6 on 2019-10-20 01:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0005_auto_20191019_1544'),
    ]

    operations = [
        migrations.CreateModel(
            name='AlphaVantageHistory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('ticker', models.CharField(max_length=12)),
                ('close_price', models.FloatField()),
                ('updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
