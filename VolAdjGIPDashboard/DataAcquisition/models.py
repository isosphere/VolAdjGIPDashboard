import datetime
import logging
import math

from django.db import models
from django.conf import settings
import pandas_datareader.data as web
import pandas as pd
import numpy as np

class SecurityHistory(models.Model):
    date = models.DateField()
    ticker = models.CharField(max_length=12)
    close_price = models.FloatField()

    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        logger = logging.getLogger('SecurityHistory.update')
        logger.setLevel(settings.LOG_LEVEL)

        end = end if end is not None else datetime.datetime.now()
        logger.info(f"Final date of interest for update: {end}")
        if start is None and not clobber:
            logger.info(f"start not specified with clobber mode disabled, will update since last record in database.")

        if tickers is None:
            tickers = cls.objects.all().values_list('ticker', flat=True).distinct()
            logger.info(f"No ticker specified, so using all distinct tickers in the database: {tickers}")
        
        for security in tickers:
            logger.info(f"Updating {security}...")

            if start is None and not clobber:
                try:
                    start = cls.objects.filter(ticker=security).latest('date').date
                except cls.DoesNotExist:
                    pass
            
            dataframe = web.DataReader(security, 'yahoo', start, end)
            
            # take credit for dividends!
            if 'Adj Close' in dataframe.columns:
                dataframe = dataframe.drop('Close', axis=1).rename({"Adj Close": "Close"}, axis=1)

            if clobber:
                cls.objects.filter(date__gte=start, date__lte=end).delete()
            
            for row in dataframe.itertuples():
                date, close_price = row.Index, row.Close
                obj, created = cls.objects.get_or_create(date=date, ticker=security, defaults={'close_price':close_price})
                obj.close_price = close_price
                obj.save()

    @classmethod
    def equal_volatility_position(cls, tickers, lookback=252, target_value=10000, max_date=None):
        logger = logging.getLogger('SecurityHistory.equal_volatility_position')
        logger.setLevel(settings.LOG_LEVEL)

        standard_move = dict()
        last_price_lookup = dict()

        controlling_leg = None
        max_price = None

        for security in tickers:
            if max_date is None:
                results = cls.objects.filter(ticker=security).order_by('-date')[:lookback+1].values('date', 'close_price')
            else:
                results = cls.objects.filter(ticker=security, date__lte=max_date).order_by('-date')[:lookback+1].values('date', 'close_price')

            if not results:
                raise ValueError(f'Need data for {security}')

            dataframe = pd.DataFrame.from_records(results, columns=['date', 'close_price'], index='date', coerce_float=True)
            dataframe.index = pd.to_datetime(dataframe.index)
            dataframe.sort_index(inplace=True, ascending=True)
           
            # compute realized vol
            dataframe["log_return"] = np.log(dataframe.close_price) - np.log(dataframe.close_price.shift(1))
            dataframe["realized_vol"] = dataframe.log_return.rolling(lookback).std(ddof=0)

            latest_close, realized_vol = dataframe.iloc[-1].close_price, dataframe.iloc[-1].realized_vol
            
            standard_move[security] = realized_vol*latest_close
            last_price_lookup[security] = latest_close

            if max_price is None or latest_close > max_price:
                controlling_leg = security
                max_price = latest_close

        logger.info(f"Controlling leg (most expensive) is {controlling_leg}.")

        leg_ratios = dict()
        for leg in set(tickers).symmetric_difference({controlling_leg}):
            leg_ratios[leg] = standard_move[controlling_leg] / standard_move[leg]
        
        base_cost = last_price_lookup[controlling_leg]
        for leg in leg_ratios:
            base_cost += last_price_lookup[leg] * leg_ratios[leg]

        multiplier = target_value // base_cost
        
        positioning = dict()
        positioning[controlling_leg] = int(multiplier)
        for leg in leg_ratios:
            positioning[leg] = math.floor(multiplier*leg_ratios[leg])
        
        return positioning

    @classmethod
    def quarter_return(cls, tickers, date_within_quarter):
        date_within_quarter += pd.offsets.QuarterEnd()*0 # this is now the quarter end date
        
        start_date = date_within_quarter - pd.offsets.QuarterEnd() + datetime.timedelta(days=1)

        positioning = cls.equal_volatility_position(tickers, max_date=date_within_quarter - pd.offsets.QuarterEnd())

        start_market_value = 0
        end_market_value = 0

        # we're assuming the positioning stays constant, which is wrong
        for leg in positioning:
            start_market_value += positioning[leg]*cls.objects.filter(ticker=leg, date__gte=start_date).earliest('date').close_price
            end_market_value += positioning[leg]*cls.objects.filter(ticker=leg, date__lte=date_within_quarter).latest('date').close_price
        
        return end_market_value / start_market_value - 1

    class Meta:
        unique_together = [['ticker', 'date']]
