from django.core.management.base import BaseCommand
from DataAcquisition.models import YahooHistory, QuadForecasts, QuadReturn, CPIForecast, GDPForecast

class Command(BaseCommand):
    help = 'Update security history and forecasts'

    def handle(self, *args, **options):
        CPIForecast.update()
        GDPForecast.update()
        YahooHistory.update()
        YahooHistory.calculate_stats()
        QuadReturn.update()

        QuadForecasts.update()
