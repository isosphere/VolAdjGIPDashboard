from django.contrib import admin
from .models import CoinGeckoPair

@admin.register(CoinGeckoPair)
class CoinGeckoPairAdmin(admin.ModelAdmin):
    pass