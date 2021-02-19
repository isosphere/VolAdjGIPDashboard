from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory, QuadForecasts, CPIForecast, BitfinexHistory

class Command(BaseCommand):
    help = 'Update security history and forecasts'

    def handle(self, *args, **options):
        CPIForecast.update()
        YahooHistory.update()
        YahooHistory.calculate_stats()
        AlphaVantageHistory.update()
        AlphaVantageHistory.calculate_stats()
        BitfinexHistory.update()
        BitfinexHistory.calculate_stats()

        QuadForecasts.update()
