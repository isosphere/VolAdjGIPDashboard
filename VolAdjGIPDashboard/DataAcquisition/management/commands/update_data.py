from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, QuadForecasts, CPIForecast, GDPForecast

class Command(BaseCommand):
    help = 'Update security history and forecasts'

    def handle(self, *args, **options):
        CPIForecast.update()
        GDPForecast.update()
        YahooHistory.update()
        YahooHistory.calculate_stats()

        QuadForecasts.update()
