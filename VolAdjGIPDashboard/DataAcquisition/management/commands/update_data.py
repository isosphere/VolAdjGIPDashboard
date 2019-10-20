from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory

import os

class Command(BaseCommand):
    help = 'Update security history'

    def handle(self, *args, **options):
        YahooHistory.update()
        AlphaVantageHistory.update()

