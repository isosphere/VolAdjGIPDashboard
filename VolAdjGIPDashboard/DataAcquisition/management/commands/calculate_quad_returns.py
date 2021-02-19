from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory, BitfinexHistory


class Command(BaseCommand):
    help = 'Update quad return calculations for all available data'

    def handle(self, *args, **options):
        YahooHistory.update_quad_return(tickers=[['QQQ',],['XLF', 'XLI', 'QQQ'],['GLD',],['XLU', 'TLT', 'UUP'],['VTI']])
        YahooHistory.update_quad_return()
        AlphaVantageHistory.update_quad_return()
        BitfinexHistory.update_quad_return()