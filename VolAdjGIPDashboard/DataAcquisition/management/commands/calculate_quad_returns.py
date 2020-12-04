from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import QuadReturn


class Command(BaseCommand):
    help = 'Update quad return calculations for all available data'

    def handle(self, *args, **options):
        QuadReturn.update() 