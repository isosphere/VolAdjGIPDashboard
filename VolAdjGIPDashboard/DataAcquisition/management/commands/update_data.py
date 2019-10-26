from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory, QuadForecasts

import os

class Command(BaseCommand):
    help = 'Update security history and forecasts'

    def handle(self, *args, **options):
        YahooHistory.update()
        AlphaVantageHistory.update()
        QuadForecasts.update()
