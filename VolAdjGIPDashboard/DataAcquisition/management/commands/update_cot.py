from django.core.management.base import BaseCommand, CommandError

from DataAcquisition.models import CommitmentOfTraders

class Command(BaseCommand):
    help = 'Update commitment of traders data'

    def handle(self, *args, **options):
        CommitmentOfTraders.update()