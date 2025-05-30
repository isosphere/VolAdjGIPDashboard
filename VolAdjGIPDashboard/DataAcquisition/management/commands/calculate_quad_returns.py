from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import YahooHistory


class Command(BaseCommand):
    help = 'Update quad return calculations for all available data'

    def handle(self, *args, **options):
        YahooHistory.update_quad_return(tickers=[['QQQ',],['XLF', 'XLI', 'QQQ'],['GLD','VPU'],['VPU', 'TLT', 'UUP'],['VTI'],['VPU','TLT','UUP','GLD']])
        YahooHistory.update_quad_return()