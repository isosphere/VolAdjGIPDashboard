# Generated by Django 2.2.6 on 2019-11-12 01:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0007_quadforecasts'),
    ]

    operations = [
        migrations.RenameField(
            model_name='quarterreturn',
            old_name='quarter_return',
            new_name='quad_return',
        ),
    ]
