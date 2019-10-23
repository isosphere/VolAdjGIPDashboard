import datetime
import logging
import json
import math
import requests

from dateutil.parser import parse

from django.db import models
from django.conf import settings
import pandas_datareader.data as web
import pandas as pd
import numpy as np

class QuarterReturn(models.Model):
    quarter_end_date = models.DateField()
    data_end_date = models.DateField()
    label = models.CharField(max_length=100)

    prices_updated = models.DateTimeField() # data taken from YahooHistory.updated

    quarter_return = models.FloatField()

    class Meta:
        unique_together = [['quarter_end_date', 'data_end_date', 'label']]
    

class SecurityHistory(models.Model):
    date = models.DateField()
    ticker = models.CharField(max_length=12)
    close_price = models.FloatField()
    updated = models.DateTimeField(auto_now=True)

    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        pass

    @classmethod
    def dataframe(cls, max_date=None, tickers=None, lookback=None):
        results = cls.objects.all().order_by('-date')
        if tickers is not None:
            results = results.filter(ticker__in=tickers)

        if max_date is not None:
            results = results.filter(date__lte=max_date)
        else:
            max_date = results.latest('date').date
        
        if lookback:
            results = results.filter(date__gte=max_date - datetime.timedelta(days=lookback*1.6)) # this math is impercise because of weekends
      
        results = results.values('date', 'ticker', 'close_price')

        dataframe = pd.DataFrame.from_records(results, columns=['date', 'ticker', 'close_price'], coerce_float=True)
        dataframe.date = pd.to_datetime(dataframe.date)
        dataframe.set_index(['ticker', 'date'], inplace=True)
        dataframe.sort_index(inplace=True, ascending=True, level=['ticker', 'date'])        

        return dataframe

    class Meta:
        abstract = True


class AlphaVantageHistory(SecurityHistory):
    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        logger = logging.getLogger('AlphaVantageHistory.update')
        logger.setLevel(settings.LOG_LEVEL)

        if clobber is True or start is not None or end is not None:
            logger.warning("Clobber, start, and end are not currently supported.")

        if tickers is None:
            tickers = cls.objects.all().values_list('ticker', flat=True).distinct()
            logger.info(f"No ticker specified, so using all distinct tickers in the database: {tickers}")

        for ticker in tickers:
            from_currency, to_currency = ticker.split('.')

            base_url = r"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
            request_url = f"{base_url}&from_currency={from_currency}&to_currency={to_currency}&apikey={settings.ALPHAVENTAGE_KEY}"

            response = requests.get(request_url)
            results = response.json()['Realtime Currency Exchange Rate']

            updated, exchange_rate = parse(results['6. Last Refreshed']), results['5. Exchange Rate']

            obj, created = cls.objects.get_or_create(date=updated.date(), ticker=ticker, defaults={'close_price':exchange_rate, 'updated': updated})
            obj.close_price = exchange_rate
            obj.realized_volatility = None # we'll calculate this later
            obj.updated = updated
            obj.save()

class YahooHistory(SecurityHistory):
    realized_volatility = models.FloatField(null=True) 

    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        logger = logging.getLogger('YahooHistory.update')
        logger.setLevel(settings.LOG_LEVEL)

        end = end if end is not None else datetime.datetime.now()
        logger.info(f"Final date of interest for update: {end}")
        if start is None and not clobber:
            logger.info(f"start not specified with clobber mode disabled, will update since last record in database.")

        if start is None and clobber:
            start = datetime.date(2008, 1, 1)

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
                cls.objects.filter(date__gte=start, date__lte=end, ticker=security).delete()
            
            for row in dataframe.itertuples():
                date, close_price = row.Index, row.Close
                obj, created = cls.objects.get_or_create(date=date, ticker=security, defaults={'close_price':close_price, 'updated': datetime.datetime.now()})
                obj.close_price = close_price
                obj.realized_volatility = None # we'll calculate this later
                obj.updated = datetime.datetime.now()
                obj.save()

    @classmethod
    def equal_volatility_position(cls, tickers, lookback=252, target_value=10000, max_date=None):
        logger = logging.getLogger('YahooHistory.equal_volatility_position')
        logger.setLevel(settings.LOG_LEVEL)

        standard_move = dict()
        last_price_lookup = dict()

        controlling_leg = None
        max_price = None

        dataframe = cls.dataframe(max_date=max_date, tickers=tickers, lookback=lookback)

        # compute realized vol
        dataframe["log_return"] = dataframe.groupby(level='ticker').close_price.apply(np.log) - dataframe.groupby(level='ticker').close_price.shift(1).apply(np.log)
        dataframe["realized_vol"] = dataframe.groupby(level='ticker').log_return.rolling(lookback).std(ddof=0).droplevel(0)

        for security in tickers:
            subset = dataframe[dataframe.index.get_level_values('ticker') == security]
            latest_close, realized_vol = subset.iloc[-1].close_price, subset.iloc[-1].realized_vol
            
            standard_move[security] = realized_vol*latest_close
            last_price_lookup[security] = latest_close

            if max_price is None or latest_close > max_price:
                controlling_leg = security
                max_price = latest_close

        logger.debug(f"Controlling leg (most expensive) is {controlling_leg}, with a standard move of ${standard_move[controlling_leg]:2f}.")

        leg_ratios = dict()
        for leg in set(tickers).symmetric_difference({controlling_leg}):
            leg_ratios[leg] = standard_move[controlling_leg] / standard_move[leg]
            logger.debug(f"leg={leg}, standard move=${standard_move[leg]:2f} ratio={leg_ratios[leg]}")
        
        base_cost = last_price_lookup[controlling_leg]
        for leg in leg_ratios:
            base_cost += last_price_lookup[leg] * leg_ratios[leg]

        logger.debug(f"Base cost: ${base_cost:.2f}")

        multiplier = target_value // base_cost
        
        positioning = dict()
        positioning[controlling_leg] = int(multiplier)
        actual_cost = int(multiplier)*last_price_lookup[controlling_leg]
        for leg in leg_ratios:
            positioning[leg] = math.floor(multiplier*leg_ratios[leg])
            actual_cost += positioning[leg]*last_price_lookup[leg]

        logger.debug(f"Actual cost: ${actual_cost:.2f}")
        
        return positioning

    @classmethod
    def quarter_return(cls, tickers, date_within_quarter):
        tickers.sort() # make the list deterministic for the same input (used for label later)

        date_within_quarter += pd.offsets.QuarterEnd()*0 # this is now the quarter end date
        
        start_date = date_within_quarter - pd.offsets.QuarterEnd() + datetime.timedelta(days=1)

        history = cls.objects.filter(ticker__in=tickers, date__gte=start_date, date__lte=date_within_quarter).order_by('date')

        try:
            cached = QuarterReturn.objects.filter(
                quarter_end_date = date_within_quarter, 
                data_end_date = date_within_quarter, 
                label = ','.join(tickers).upper(),
                prices_updated__gte = history.latest('updated').updated
            ).latest('prices_updated')

            return cached.quarter_return

        except QuarterReturn.DoesNotExist:
            pass

        distinct_dates = history.values('date').distinct().values_list('date', flat=True)
        
        prior_positioning = None
        prior_cost_basis = dict()
        start_market_value = 10000
        market_value = start_market_value
        
        for date in distinct_dates:
            # liquidate
            if prior_positioning is not None:
                for leg in prior_positioning:
                    market_value += prior_positioning[leg]*(history.get(ticker=leg, date=date).close_price - prior_cost_basis[leg])
                    
            # accumulate
            new_positioning = cls.equal_volatility_position(tickers, max_date=date, target_value=market_value)
            
            prior_cost_basis = dict()
            prior_positioning = new_positioning.copy()

            for leg in new_positioning:
                prior_cost_basis[leg] = history.get(ticker=leg, date=date).close_price
        
        end_market_value = market_value

        quarter_return = end_market_value / start_market_value - 1
        
        QuarterReturn.objects.filter(
            quarter_end_date = date_within_quarter, 
            data_end_date = date_within_quarter, 
            label = ','.join(tickers).upper(),
        ).delete() # if we had an older one, kill it
        
        cached_return = QuarterReturn(
            quarter_end_date = date_within_quarter, 
            data_end_date = date_within_quarter, 
            label = ','.join(tickers).upper(),
            prices_updated = history.latest('updated').updated,
            quarter_return = quarter_return 
        )
        cached_return.save()
        
        return end_market_value / start_market_value - 1

    class Meta:
        unique_together = [['ticker', 'date']]
