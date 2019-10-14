# Generated by Django 2.2.6 on 2019-10-14 21:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SecurityHistory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('ticker', models.CharField(max_length=12)),
                ('close_price', models.FloatField()),
            ],
            options={
                'unique_together': {('ticker', 'date')},
            },
        ),
    ]
