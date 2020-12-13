# Generated by Django 2.2.13 on 2020-12-05 00:28

from django.db import migrations

def forwards_func(apps, schema_edtior):
    #YahooHistory = apps.get_model("DataAcquisition", "YahooHistory")
    #QuadReturn = apps.get_model("DataAcquisition", "QuadReturn")
    from DataAcquisition.models import YahooHistory, QuadReturn # bad practice, but only way to access class methods

    print("Force updating YahooHistory symbols ...")
    YahooHistory.update(tickers=['QQQ', 'XLI', 'XLF', 'GLD', 'XLU', 'UUP', 'TLT'], clobber=True)

    print("Wiping out QuadReturn data ...")
    QuadReturn.objects.all().delete()

    print("Recalculating QuadReturn data ...")
    QuadReturn.update()


class Migration(migrations.Migration):

    dependencies = [
        ('DataAcquisition', '0018_remove_yahoohistory_realized_volatility_week'),
    ]

    operations = [
        migrations.RunPython(forwards_func, migrations.RunPython.noop)
    ]