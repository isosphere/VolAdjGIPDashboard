from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory, AlphaVantageHistory, BitfinexHistory, CoinGeckoHistory


class Command(BaseCommand):
    help = 'Update quad return calculations for all available data'

    def handle(self, *args, **options):
        YahooHistory.update_quad_return(tickers=[['QQQ',],['XLF', 'XLI', 'QQQ'],['GLD','VPU'],['VPU', 'TLT', 'UUP'],['VTI'],['VPU','TLT','UUP','GLD']])
        YahooHistory.update_quad_return()
        AlphaVantageHistory.update_quad_return()
        BitfinexHistory.update_quad_return()
        CoinGeckoHistory.update_quad_return()