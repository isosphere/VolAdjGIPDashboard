from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory, QuadForecasts, CPIForecast, BitfinexHistory, CoinGeckoHistory, GDPForecast

class Command(BaseCommand):
    help = 'Update security history and forecasts'

    def handle(self, *args, **options):
        CPIForecast.update()
        GDPForecast.update()
        YahooHistory.update()
        YahooHistory.calculate_stats()
        BitfinexHistory.update()
        BitfinexHistory.calculate_stats()
        CoinGeckoHistory.update()
        CoinGeckoHistory.calculate_stats()

        QuadForecasts.update()
