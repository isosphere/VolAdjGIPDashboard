# Generated by Django 2.2.13 on 2021-02-19 15:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0021_QuadReturn_generalization'),
    ]

    operations = [
        migrations.CreateModel(
            name='BitfinexHistory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('ticker', models.CharField(max_length=12)),
                ('close_price', models.FloatField()),
                ('updated', models.DateTimeField(auto_now=True)),
                ('realized_volatility', models.FloatField(null=True)),
            ],
            options={
                'unique_together': {('ticker', 'date')},
            },
        ),
    ]
