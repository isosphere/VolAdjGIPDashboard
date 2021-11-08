from django.core.management.base import BaseCommand, CommandError
from django.core.cache import cache

class Command(BaseCommand):
    help = 'Clear Django cache'

    def handle(self, *args, **options):
        cache.clear()
        print("Cache cleared.")
