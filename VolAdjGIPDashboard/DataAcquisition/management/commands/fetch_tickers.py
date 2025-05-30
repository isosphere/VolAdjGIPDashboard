from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Fetch data for specific tickers'

    def add_arguments(self, parser):
        parser.add_argument('ticker', type=str, nargs="+")

    def handle(self, *args, **options):
        from DataAcquisition.models import YahooHistory, QuadReturn

        if isinstance(options['ticker'], str):
            tickers = [ options['ticker'].upper() ]
        else:
            tickers = [ x.upper() for x in options['ticker'] ]
        
        YahooHistory.update(tickers=tickers, clobber=True)   
        YahooHistory.calculate_stats()

        tickers = [ [ x ] for x in tickers ]

        YahooHistory.update_quad_return(tickers=tickers, full_run=True)
        QuadReturn.update()