from django.core.management.base import BaseCommand
from DataAcquisition.models import YahooHistory, QuadForecasts, QuadReturn, CPIForecast, GDPForecast

class Command(BaseCommand):
    help = 'Update security history and forecasts. This command is sufficient for an empty database, or an out-of-date one.'

    def handle(self, *args, **options):
        CPIForecast.update()
        GDPForecast.update()
        YahooHistory.update()
        YahooHistory.calculate_stats()
        YahooHistory.update_quad_return()
        QuadReturn.update()

        QuadForecasts.update()
