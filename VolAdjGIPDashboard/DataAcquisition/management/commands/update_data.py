from django.core.management.base import BaseCommand, CommandError
from DataAcquisition.models import SecurityHistory

import os

class Command(BaseCommand):
    help = 'Update security history'

    def handle(self, *args, **options):
        SecurityHistory.update()
