from django.core.management.base import BaseCommand, CommandError
from django.core.cache import cache

class Command(BaseCommand):
    help = 'Fetch data for a specific ticker'

    def add_arguments(self, parser):
        parser.add_argument('ticker', type=str)

    def handle(self, *args, **options):
        from DataAcquisition.models import YahooHistory
        
        YahooHistory.update(tickers=(options['ticker'],))   
        YahooHistory.calculate_stats()
        YahooHistory.update_quad_return(ticker=options['ticker'])