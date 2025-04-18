import asyncio
import datetime
import logging
import io
import itertools
import math
import requests
import time

from dateutil.parser import parse
import pytz

from django.db import models
from django.db.models import Max
from django.db.utils import IntegrityError
from django.conf import settings

from bfxapi import Client
import pandas_datareader.data as web
import yfinance
import quandl

import pandas as pd
import numpy as np

class QuadReturn(models.Model):
    quarter_end_date = models.DateField()
    data_start_date = models.DateField()
    data_end_date = models.DateField()
    label = models.CharField(max_length=100)

    prices_updated = models.DateTimeField() # data taken from YahooHistory.updated

    quad_return = models.FloatField()
    quad_stdev = models.FloatField()

    class Meta:
        unique_together = [['quarter_end_date', 'data_start_date', 'data_end_date', 'label']]


class SecurityHistory(models.Model):
    date = models.DateField()
    ticker = models.CharField(max_length=12)
    close_price = models.FloatField()
    updated = models.DateTimeField(auto_now=True)
    realized_volatility = models.FloatField(null=True) 

    @classmethod
    def quad_return(cls, tickers, date_within_quad):
        logger = logging.getLogger('SecurityHistory.quad_return')
        tickers.sort() # make the list deterministic for the same input (used for label later)

        quarter_end_date = (date_within_quad + pd.tseries.offsets.QuarterEnd(n=0)).date()
        current_quad = QuadForecasts.objects.filter(quarter_end_date=quarter_end_date).latest('date')
        logger.debug("current_quad quarter=%s date=%s", current_quad.quarter_end_date, current_quad.date)

        # this is the last known date for the prior quad
        start_date = (date_within_quad - pd.tseries.offsets.QuarterEnd(1) + datetime.timedelta(days=1)).date()
        
        logger.debug("last known date for prior quad: %s", start_date)
        
        # this is when we started this quad
        history = cls.objects.filter(ticker__in=tickers, date__gte=start_date, date__lte=date_within_quad).order_by('date')
        end_date = date_within_quad if date_within_quad < history.latest('date').date else history.latest('date').date

        try:
            cached = QuadReturn.objects.filter(
                quarter_end_date = current_quad.quarter_end_date,
                data_start_date = start_date,
                data_end_date = end_date, 
                label = cls.__name__ + '_' + ','.join(tickers).upper(),
                prices_updated__gte = history.latest('updated').updated
            ).latest('prices_updated')

            return cached.quad_return, cached.quad_stdev

        except QuadReturn.DoesNotExist:
            pass

        distinct_dates = history.values('date').distinct().values_list('date', flat=True)
        
        prior_positioning = None
        prior_cost_basis = dict()
        start_market_value = 1e7
        market_value = start_market_value

        market_value_history = [start_market_value,]
        
        for date in distinct_dates:
            # liquidate
            if prior_positioning is not None:
                for leg in prior_positioning:
                    market_value += prior_positioning[leg]*(history.get(ticker=leg, date=date).close_price - prior_cost_basis[leg])
            
            market_value_history.append(market_value)

            # accumulate
            new_positioning = cls.equal_volatility_position(tickers, max_date=date, target_value=market_value)
            
            prior_cost_basis = dict()
            prior_positioning = new_positioning.copy()

            for leg in new_positioning:
                prior_cost_basis[leg] = history.get(ticker=leg, date=date).close_price
        
        end_market_value = market_value

        quad_return = end_market_value / start_market_value - 1
        quad_stdev = pd.DataFrame(market_value_history).pct_change().std(ddof=1).values[0]
        quad_stdev = quad_stdev if np.isfinite(quad_stdev) else 1.0
        
        QuadReturn.objects.filter(
            quarter_end_date = current_quad.quarter_end_date,
            data_start_date = start_date, 
            data_end_date = end_date, 
            label = cls.__name__ + '_' + ','.join(tickers).upper(),
        ).delete() # if we had an older one, kill it
        
        cached_return = QuadReturn(
            quarter_end_date = current_quad.quarter_end_date, 
            data_end_date = end_date,
            data_start_date = start_date,
            label = cls.__name__ + '_' + ','.join(tickers).upper(),
            prices_updated = history.latest('updated').updated,
            quad_return = quad_return,
            quad_stdev = quad_stdev 
        )
        cached_return.save()
        
        return quad_return, quad_stdev

    @classmethod
    def core_tickers(cls):
        return list(map(lambda x: [x.upper()], cls.objects.values_list('ticker', flat=True).distinct()))

    @classmethod
    def update_quad_return(cls, first_date=None, ticker=None, tickers=None, full_run=False):
        logger = logging.getLogger('SecurityHistory.update_quad_return')
        
        if ticker is None and tickers is None:
            tickers = cls.core_tickers()
        elif ticker is not None:
            tickers = [[ticker,]]

        latest_date = cls.objects.latest('date').date
        first_date = first_date if first_date is not None else cls.objects.earliest('date').date

        logging.info("Latest date of data = %s.", latest_date)
        for labels in tickers:
            sortable = labels
            sortable.sort()
            modified_label =  cls.__name__ + "_" + ','.join(sortable)

            try:
                existing_data_start = QuadReturn.objects.filter(label=modified_label).latest('data_end_date')
            except QuadReturn.DoesNotExist:
                existing_data_start = None
            
            if not full_run:
                try_date = first_date if not existing_data_start else existing_data_start.data_end_date
            else:
                try_date = first_date
            
            logger.debug("Calculating quad returns since %s for tickers %s", try_date, labels)

            while try_date <= latest_date:
                logger.debug(f"Tickers=%s, date=%s ... ", labels, try_date)
                try:
                    cls.quad_return(
                        tickers=labels,
                        date_within_quad=try_date
                    )
                    logger.debug("Ok.")

                    try_date += datetime.timedelta(days=1)

                except cls.DoesNotExist:
                    logger.debug("No data instance for that date.")
                    try_date += datetime.timedelta(days=1)

                except QuadForecasts.DoesNotExist:
                    logger.debug("No QuadForecast instance for that date.")
                    try_date += datetime.timedelta(days=1)
                
                except IntegrityError:
                    logger.debug("IntegrityError saving calculated return.")
                    try_date += datetime.timedelta(days=1)

                except ValueError:
                    logger.debug("Insufficient data for calculation.")
                    try_date += datetime.timedelta(days=1)

    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        pass

    @classmethod
    def equal_volatility_position(cls, tickers, lookback=28, target_value=10000, max_date=None):
        logger = logging.getLogger('SecurityHistory.equal_volatility_position')
        logger.debug("function triggered for tickers=[%s], lookback=%s, target_value=%s, max_date=%s", tickers, lookback, target_value, max_date)

        standard_move = dict()
        last_price_lookup = dict()

        controlling_leg = None
        max_price = None

        dataframe = cls.dataframe(max_date=max_date, tickers=tickers, lookback=lookback)

        # compute realized vol
        dataframe["log_return"] = dataframe.groupby(level='ticker', group_keys=False).close_price.apply(np.log) - dataframe.groupby(level='ticker', group_keys=False).close_price.shift(1).apply(np.log)
        dataframe["realized_vol"] = dataframe.groupby(level='ticker', group_keys=False).log_return.rolling(lookback).std(ddof=0).droplevel(0)

        for security in tickers:
            subset = dataframe[dataframe.index.get_level_values('ticker') == security].dropna()
            if subset.empty:
                logger.error("No data for one of the legs ('%s', max date %s), skipping.", security, max_date)
                raise ValueError

            latest_close, realized_vol = subset.iloc[-1].close_price, subset.iloc[-1].realized_vol
            logger.debug("%s close=%s, realized_vol=%s", security, latest_close, realized_vol)
            
            standard_move[security] = realized_vol*latest_close
            last_price_lookup[security] = latest_close

            if max_price is None or latest_close > max_price:
                controlling_leg = security
                max_price = latest_close

        logger.debug("Controlling leg (most expensive) is %s, with a standard move of $%2f.", controlling_leg, standard_move[controlling_leg])

        leg_ratios = dict()
        for leg in set(tickers).symmetric_difference({controlling_leg}):
            leg_ratios[leg] = standard_move[controlling_leg] / standard_move[leg]
            logger.debug("leg=%s, standard move=$%2f ratio=%s", leg, standard_move[leg], leg_ratios[leg])
        
        base_cost = last_price_lookup[controlling_leg]
        for leg in leg_ratios:
            base_cost += last_price_lookup[leg] * leg_ratios[leg]

        logger.debug("Base cost: $%2f", base_cost)

        multiplier = target_value // base_cost
        
        positioning = dict()
        positioning[controlling_leg] = int(multiplier)
        actual_cost = int(multiplier)*last_price_lookup[controlling_leg]
        for leg in leg_ratios:
            positioning[leg] = math.floor(multiplier*leg_ratios[leg])
            actual_cost += positioning[leg]*last_price_lookup[leg]

        logger.debug("Actual cost: $%2f", actual_cost)
        
        return positioning

    @classmethod
    def daily_return(cls, tickers):
        date_set = cls.objects.order_by('-date').values_list('date', flat=True).distinct()[:2] # latest two dates
        history = cls.objects.filter(ticker__in=tickers, date__in=date_set).order_by('date')

        prior_positioning = None
        prior_cost_basis = dict()
        start_market_value = 10000
        market_value = start_market_value

        market_value_history = [start_market_value,]
        
        for date in date_set:
            # liquidate
            if prior_positioning is not None:
                for leg in prior_positioning:
                    try:
                        market_value += prior_positioning[leg]*(history.get(ticker=leg, date=date).close_price - prior_cost_basis[leg])
                    except cls.DoesNotExist:
                        return None

            market_value_history.append(market_value)

            # accumulate
            new_positioning = cls.equal_volatility_position(tickers, max_date=date, target_value=market_value)
            
            prior_cost_basis = dict()
            prior_positioning = new_positioning.copy()

            for leg in new_positioning:
                prior_cost_basis[leg] = history.get(ticker=leg, date=date).close_price
        
        end_market_value = market_value

        daily_return = end_market_value / start_market_value - 1

        return daily_return

    @classmethod
    def weekly_return(cls, tickers):
        logger = logging.getLogger('SecurityHistory.weekly_return')

        date_set = cls.objects.order_by('-date').values_list('date', flat=True).distinct()
        history = cls.objects.filter(ticker__in=tickers, date__in=date_set).order_by('date')

        latest_date = date_set.latest('date')
        year, prior_weeknum, _ = (latest_date - datetime.timedelta(days=7)).isocalendar()
        prior_week_close_date = date_set.filter(date__week=prior_weeknum, date__year=year).latest('date')

        prior_positioning = None
        prior_cost_basis = dict()
        start_market_value = 10000
        market_value = start_market_value

        market_value_history = [start_market_value,]
        
        for date in (prior_week_close_date, latest_date):
            # liquidate
            if prior_positioning is not None:
                for leg in prior_positioning:
                    if leg not in prior_positioning:
                        logger.error("%s is not in prior_positioning, skipping", leg)
                        continue
                    if leg not in prior_cost_basis:
                        logger.error("%s is not in prior_cost_basis, skipping", leg)
                        continue
                    try:
                        market_value += prior_positioning[leg]*(history.get(ticker=leg, date=date).close_price - prior_cost_basis[leg])
                    except cls.DoesNotExist:
                        logger.error("Unable to get history for ticker %s for date %s", leg, date)
                        continue

            market_value_history.append(market_value)

            # accumulate
            new_positioning = cls.equal_volatility_position(tickers, max_date=date, target_value=market_value)
            
            prior_cost_basis = dict()
            prior_positioning = new_positioning.copy()

            for leg in new_positioning:
                try:
                    prior_cost_basis[leg] = history.get(ticker=leg, date=date).close_price
                except cls.DoesNotExist:
                    logger.error("Unable to get history for ticker %s for date %s", leg, date)
                    continue
        
        end_market_value = market_value

        weekly_return = end_market_value / start_market_value - 1

        return weekly_return

    @classmethod
    def dataframe(cls, max_date = None, tickers = None, lookback = None) -> pd.DataFrame:
        results = cls.objects.all().order_by('-date')

        if not results.exists():
            return pd.DataFrame()

        if tickers is not None:
            results = results.filter(ticker__in=tickers)

        if max_date is not None:
            results = results.filter(date__lte=max_date)
        else:
            try:
                max_date = results.latest('date').date
            except cls.DoesNotExist:
                pass
        
        if lookback:
            results = results.filter(date__gte=max_date - datetime.timedelta(days=lookback*2)) # this math is impercise because of weekends
        
        results = results.values('date', 'ticker', 'close_price')

        dataframe = pd.DataFrame.from_records(results, columns=['date', 'ticker', 'close_price'], coerce_float=True)
        dataframe.date = pd.to_datetime(dataframe.date)
        dataframe.set_index(['ticker', 'date'], inplace=True)
        dataframe.sort_index(inplace=True, ascending=True, level=['ticker', 'date'])

        return dataframe

    @classmethod
    def calculate_stats(cls, lookback=52):
        logger = logging.getLogger('SecurityHistory.calculate_stats')
        
        missing_sections = set(cls.objects.filter(realized_volatility__isnull=True).values_list('date', 'ticker').distinct())
        
        # calculate all
        all_data = cls.dataframe()
        
        if all_data.empty:
            logger.warn("Can't calculate stats when we have no data.")
            return
        
        all_data = all_data.groupby([
            pd.Grouper(level='ticker'),
            pd.Grouper(level='date', freq='W')
        ], group_keys=False).last().apply(np.log)

        all_data['prior'] = all_data.groupby('ticker', group_keys=False).close_price.shift(1)
        all_data = (all_data.close_price - all_data.prior).groupby('ticker', group_keys=False).rolling(lookback).std(ddof=0).droplevel(0).dropna()

        for date, ticker in missing_sections:
            weekending = (date + pd.offsets.Week(weekday=6)).date()

            result = all_data.loc[
                (all_data.index.get_level_values('date') == "%s" % weekending) &
                (all_data.index.get_level_values('ticker') == ticker)
            ]

            if not result.empty:
                cls.objects.filter(ticker=ticker, date=date).update(realized_volatility=result.values[0])
    
    @classmethod
    def add_tickers(cls, tickers=[]):
        cls.update(tickers=tickers)
        cls.calculate_stats()
        cls.update_quad_return(tickers=[[x] for x in tickers])

    def __str__(self):
        return f"{self.ticker} on {self.date} was {self.close_price}"

    class Meta:
        abstract = True


class BitfinexHistory(SecurityHistory):
    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        logger = logging.getLogger('BitfinexHistory.update')

        bfx = Client()

        if clobber is True:
            logger.warning("Clobber not currently supported.")

        if tickers is None:
            tickers = cls.objects.all().values_list('ticker', flat=True).distinct()
            logger.info(f"No ticker specified, so using all distinct tickers in the database: {tickers}")

        if end is None:
            end = int(round(time.time() * 1000))
        else:
            end = int(round(end.timestamp() * 1000))

        if start is None:
            start = end - (1000 * 60 * 60 * 24 * 7) # 7 days ago
        else:
            start = int(round(start.timestamp() * 1000))

        runtime = datetime.datetime.now()

        for ticker in tickers:
            candles = asyncio.run(bfx.rest.get_public_candles(f't{ticker}', start, end, tf='1D', limit="10000"))
            for milli_timestamp, open, close, high, low, volume in candles:
                date = datetime.datetime.fromtimestamp(milli_timestamp/1000.0, tz=pytz.utc).date()
                obj, created = cls.objects.get_or_create(date=date, ticker=ticker, defaults={'close_price':close, 'updated': runtime})
                obj.close_price = close
                obj.realized_volatility = None # we'll calculate this later
                obj.updated = runtime
                obj.save() # it would be faster if we deferred these saves and did a bulk or atomic operation
    
    @classmethod
    def backfill(cls, tickers=None):
        end = datetime.datetime.now()
        start = end -  datetime.timedelta(days=252*7) # 7 years ago
        cls.update(tickers, start=start, end=end)

    def __str__(self):
        return f"{self.ticker} on {self.date} was {self.close_price} with 1-week vol {self.realized_volatility}"

    class Meta:
        unique_together = [['ticker', 'date']]


class YahooHistory(SecurityHistory):
    @classmethod
    def core_tickers(cls):
        tickers = list(map(lambda x: [x.upper()], cls.objects.values_list('ticker', flat=True).distinct()))
        tickers.append(['QQQ', 'XLF', 'XLI'])
        tickers.append(['TLT', 'UUP', 'VPU'])
        for ticker in ('CAD=X', 'SPY', 'DJP', 'XLE'):
            tickers.append([ticker.upper()])

        return tickers
    
    @classmethod
    def update(cls, tickers=None, clobber=False, start=None, end=None):
        logger = logging.getLogger('YahooHistory.update')

        if isinstance(tickers, str):
            tickers = [tickers]

        end = end if end is not None else datetime.date.today()
        logger.info("Final date of interest for update: %s", end)
        if start is None and not clobber:
            logger.info("start not specified with clobber mode disabled, will update since last record in database.")

        if start is None and clobber:
            start = datetime.date(2008, 1, 1)

        if tickers is None:
            tickers = set(itertools.chain(*cls.core_tickers()))
            logger.info("No ticker specified, so using all distinct tickers in the database: %s", tickers)

        if start is None and not clobber:
            existing = YahooHistory.objects.filter(ticker__in=tickers).values('ticker').distinct().annotate(Max('date')).values_list('date__max', flat=True)
            if existing.exists():
                start = min(existing)

        dataframe = yfinance.download(tickers=tickers, interval='1d', start=start, end=end + datetime.timedelta(days=1), auto_adjust=True, progress=False).resample('D').last()
        for security in tickers:
            logger.info(f"Updating {security}...")
            try:
                subdf = dataframe.loc[:, ('Close', security)].dropna()
            except KeyError:
                if len(dataframe.columns) <= 5:
                    subdf = dataframe.Close
                else:
                    logger.error(f"No data found for {security} - dataframe is empty")
                    continue

            if subdf.empty:
                logger.error(f"No data found for {security} - dataframe is empty")
                continue

            if clobber:
                cls.objects.filter(date__gte=start, date__lte=end, ticker=security).delete()
            
            for date, close_price in subdf.items():
                obj, created = cls.objects.get_or_create(date=date.date(), ticker=security, defaults={'close_price':close_price, 'updated': datetime.datetime.now()})
                obj.close_price = close_price
                obj.realized_volatility = None # we'll calculate this later
                obj.realized_volatility_week = None
                obj.updated = datetime.datetime.now()
                obj.save()

    def __str__(self):
        return f"{self.ticker} on {self.date} was {self.close_price} with 1-week vol {self.realized_volatility}"

    class Meta:
        unique_together = [['ticker', 'date']]


class GDPForecast(models.Model):
    quarter_end_date = models.DateField()
    date = models.DateField() # updated date
    gdp = models.FloatField() # y/y

    @classmethod
    def update(cls):
        url = r'http://www.forecasts.org/inf/gdp-growth-forecast-data.csv'
    
        response = requests.get(url, allow_redirects=True)

        # determine when this data was updated
        url_time_string = response.headers['last-modified']
        url_time = datetime.datetime.strptime(url_time_string, '%a, %d %b %Y %H:%M:%S %Z')

        memory_handle = io.BytesIO(response.content)
        
        dataframe = pd.read_csv(memory_handle, header=0, names=["Quarter", "GDP", "Note"], skipfooter=2, engine="python")
        dataframe = dataframe.loc[dataframe.Note.str.contains("Forecast")]
        
        dataframe["Quarter"] = pd.to_datetime(dataframe.Quarter)
        dataframe.set_index('Quarter', inplace=True)
        
        gdp_forecasts = dataframe.GDP.resample(rule='Q', level='Quarter').mean()

        for quarter, gdp in gdp_forecasts.items():
            cls.objects.update_or_create(
                quarter_end_date = quarter.date(), 
                date = url_time, 
                defaults = {
                    'gdp': gdp,
                }
            )
    
    @classmethod
    def dataframe(cls):
        latest_date = cls.objects.latest('date').date

        dataset = pd.DataFrame.from_records(cls.objects.filter(date=latest_date).values())
        dataset.quarter_end_date = pd.to_datetime(dataset.quarter_end_date)
        dataset.set_index('quarter_end_date', inplace=True)

        return dataset


class CPIForecast(models.Model):
    quarter_end_date = models.DateField()
    date = models.DateField() # updated date
    cpi = models.FloatField()

    @classmethod
    def update(cls):
        url = r'https://www.forecasts.org/inf/cpi-data.csv'
    
        response = requests.get(url, allow_redirects=True)

        # determine when this data was updated
        url_time_string = response.headers['last-modified']
        url_time = datetime.datetime.strptime(url_time_string, '%a, %d %b %Y %H:%M:%S %Z')

        memory_handle = io.BytesIO(response.content)
        
        dataframe = pd.read_csv(memory_handle, header=0, names=["Quarter", "CPI", "Note"], skipfooter=2, engine="python")
        dataframe = dataframe.loc[dataframe.Note.str.contains("Forecast")]
        
        dataframe["Quarter"] = pd.to_datetime(dataframe.Quarter)
        dataframe.set_index('Quarter', inplace=True)
        
        cpi_forecasts = dataframe.CPI.resample(rule='Q', level='Quarter').mean()

        for quarter, cpi in cpi_forecasts.items():
            cls.objects.update_or_create(
                quarter_end_date = quarter.date(), 
                date = url_time, 
                defaults = {
                    'cpi': cpi,
                }
            )
    
    @classmethod
    def dataframe(cls):
        latest_date = cls.objects.latest('date').date

        dataset = pd.DataFrame.from_records(cls.objects.filter(date=latest_date).values())
        dataset.quarter_end_date = pd.to_datetime(dataset.quarter_end_date)
        dataset.set_index('quarter_end_date', inplace=True)

        return dataset

class QuadForecasts(models.Model):
    quarter_end_date = models.DateField()
    date = models.DateField()

    cpi_roc = models.FloatField()
    gdp_roc = models.FloatField()
    quad = models.IntegerField()

    updated = models.DateTimeField(auto_now=True)

    @classmethod
    def __determine_quad_multi_index(cls, row):
        if row.cpi_roc[0] <= 0 and row.gdp_roc[0] >= 0:
            return 1
        # quad 2
        elif row.cpi_roc[0] > 0 and row.gdp_roc[0] >= 0:
            return 2
        # quad 3 
        elif row.cpi_roc[0] > 0 and row.gdp_roc[0] < 0:
            return 3
        # quad 4
        elif row.cpi_roc[0] <= 0 and row.gdp_roc[0] < 0:
            return 4

    @classmethod
    def fetch_usa_gi_data(cls, start_date=None):
        '''
        Fetches the latest GDP and CPI numbers + forecasts. Excludes older forecast data.
        '''

        if start_date is None:
            start_date = datetime.date(1950, 1, 1)

        # Real GDP, seasonally adjusted. Quarterly.
        gdp_data = web.get_data_fred('GDPC1', start = start_date)['GDPC1']

        # align the FRED quarterly dates to Pandas quarterly dates
        # each index value will be the last day of a quarter. i.e. 2019-06-30 is Q2 2019.
        gdp_data.index = gdp_data.index + pd.offsets.QuarterEnd(n=0)
        gdp_data = gdp_data.resample('Q').asfreq()

        # CPI, all items, urban, not seasonally adjusted. Monthly.
        cpi_all_urban_unadjusted_data = web.get_data_fred('CPIAUCNS', start = start_date)['CPIAUCNS']
        cpi_all_urban_unadjusted_data.index = pd.to_datetime(cpi_all_urban_unadjusted_data.index)
        cpi_data = cpi_all_urban_unadjusted_data.resample('Q').mean()

        cpi_nowcasts = CPIForecast.dataframe()
        latest_date = cpi_nowcasts.date.max()

        cpi_data = pd.concat([cpi_data[~(cpi_data.index.isin(cpi_nowcasts.cpi.index))], cpi_nowcasts.cpi])

        return gdp_data, cpi_data, latest_date

    # As of 2021-09-03, the GDP nowcast has been suspended. The pandemic caused "volatility" so they gave up.
    @classmethod
    def get_new_york_fed_gdp_nowcasts(cls):
        url = r'https://www.newyorkfed.org/medialibrary/media/research/policy/nowcast/new-york-fed-staff-nowcast_data_2002-present.xlsx?la=en'

        response = requests.get(url, allow_redirects=True)
        memory_handle = io.BytesIO(response.content)

        data = pd.read_excel(memory_handle, sheet_name='Forecasts By Quarter', header=13)
        latest_date = data['Forecast Date'].max().date()

        data = data.melt(id_vars=['Forecast Date'], var_name="quarter", value_name="forecast")
        data.columns = ['date', 'quarter', 'forecast']
        data['quarter'] = pd.to_datetime(data.quarter) + pd.offsets.QuarterEnd(n=0)*0
        
        return data, latest_date

    @classmethod
    def get_gdp_set(cls, actual_gdp):
        data = GDPForecast.dataframe() # y/y GDP. quarterly data. 
        latest_date = data.date.max()
        data['growth'] = data.gdp / 100 + 1

        gdp_df = pd.DataFrame({
            'date': actual_gdp.index + pd.offsets.QuarterEnd(n=4), # pre-shift, for easy multiplying y/y later
            'prior_actual_gdp': actual_gdp.values
        }).set_index('date')

        first_order_estimates = data.join(gdp_df, on='quarter_end_date')
        first_order_estimates['number'] = (first_order_estimates.growth * first_order_estimates.prior_actual_gdp)
        first_order_estimates.drop(['prior_actual_gdp', 'id', 'gdp'], inplace=True, axis='columns')
        first_order_estimates.reset_index(inplace=True)
        first_order_estimates.date = pd.to_datetime(first_order_estimates.date)
        first_order_estimates.set_index(['quarter_end_date', 'date'], inplace=True)

        actual_current_gdp = pd.DataFrame({
            'date': actual_gdp.index,
            'quarter_end_date': actual_gdp.index + pd.offsets.QuarterEnd(n=0),
            'actual_gdp': actual_gdp.values
        }).set_index(['quarter_end_date', 'date'])

        second_order_estimates = pd.concat(
            objs=[actual_current_gdp, first_order_estimates],
            keys=('quarter_end_date', 'date'),
            axis=0, sort=True
        )

        second_order_estimates = second_order_estimates.assign(
            best_estimate = np.where(
                second_order_estimates.actual_gdp.isnull(),
                second_order_estimates.number,
                second_order_estimates.actual_gdp
            )
        )

        second_order_estimates.drop(['number', 'actual_gdp'], inplace=True, axis='columns') # these columns are confusing anyway due to shifting
        second_order_estimates.dropna(how='all', inplace=True) # if all columns are null, drop
        
        return second_order_estimates, latest_date

    @classmethod
    def determine_quads(cls, actual_gdp, actual_cpi):
        dataframe, latest_date = cls.get_gdp_set(actual_gdp)
        dataframe.drop('growth', inplace=True, axis='columns')
        
        cpi_df = pd.DataFrame({'cpi': actual_cpi, 'quarter_end_date': actual_cpi.index + pd.offsets.QuarterEnd(n=0)}).set_index('quarter_end_date')
        dataframe = dataframe.join(cpi_df, on='quarter_end_date')

        # Collected settled YOY numbers
        dataframe.reset_index(inplace=True)
        dataframe['prior_year_quarter_end_date'] = dataframe['quarter_end_date'] - pd.offsets.QuarterEnd(n=4)

        dataframe = pd.merge(
            left=dataframe, 
            right=dataframe[['best_estimate', 'cpi', 'quarter_end_date']], 
            left_on="prior_year_quarter_end_date", 
            right_on="quarter_end_date", 
            suffixes=(None, "_prior")
        )

        dataframe['gdp_yoy'] = dataframe['best_estimate'] / dataframe['best_estimate_prior'] - 1
        dataframe['cpi_yoy'] = dataframe['cpi'] / dataframe['cpi_prior'] - 1

        dataframe['prior_quarter'] = dataframe['quarter_end_date'] - pd.offsets.QuarterEnd(n=1)
        dataframe = pd.merge(
            left=dataframe,
            right=dataframe[['gdp_yoy', 'cpi_yoy', 'quarter_end_date']],
            left_on='prior_quarter',
            right_on='quarter_end_date',
            suffixes=(None, "_priorQ")
        )

        # quarterly rate of change of the yoy rates
        dataframe['gdp_roc'] = (dataframe.gdp_yoy - dataframe.gdp_yoy_priorQ) * 1e4 # bps
        dataframe['cpi_roc'] = (dataframe.cpi_yoy - dataframe.cpi_yoy_priorQ) * 1e4 # bps

        dataframe.dropna(inplace=True)
        dataframe.set_index(['quarter_end_date', 'date'], inplace=True)
        dataframe['quad'] = dataframe.groupby(['quarter_end_date', 'date'], group_keys=False).apply(cls.__determine_quad_multi_index).rename('quad')

        # drop confusing intermediates
        dataframe.drop([
            'cpi', 'best_estimate', 'gdp_yoy', 'cpi_yoy', 'prior_year_quarter_end_date', 'prior_quarter',
            'gdp_yoy_priorQ', 'cpi_yoy_priorQ', 'best_estimate_prior', 'cpi_prior', 'quarter_end_date_priorQ',
            'quarter_end_date_prior'
        ], inplace=True, axis='columns')

        return dataframe, latest_date

    @classmethod
    def update(cls):
        gdp, cpi, latest_cpi_date = cls.fetch_usa_gi_data()
        usa_quads, latest_gdp_date = cls.determine_quads(gdp, cpi)
        usa_quads.dropna(inplace=True)

        latest_date = max(latest_cpi_date, latest_gdp_date)

        max_date = usa_quads.index.get_level_values('date').max()

        usa_quads = usa_quads[
            (usa_quads.index.get_level_values('date') <= usa_quads.index.get_level_values('quarter_end_date'))
            #&(usa_quads.index.get_level_values('date') > usa_quads.index.get_level_values('quarter_end_date') - pd.offsets.QuarterEnd(n=1))
        ]

        for row in usa_quads.itertuples():
            quarter, date = row.Index

            try:
                quad = int(row.quad)
            except ValueError:
                continue

            cls.objects.update_or_create(
                quarter_end_date = quarter.date(),
                date = latest_date,
                defaults = {
                    'updated': datetime.datetime.now(),
                    'cpi_roc': row.cpi_roc,
                    'gdp_roc': row.gdp_roc,
                    'quad': int(row.quad)
                }
            )

    class Meta:
        unique_together = [['quarter_end_date', 'date']]

class CommitmentOfTraders(models.Model):
    symbol = models.TextField()
    date = models.DateField()
    net_long = models.FloatField()
    one_year_z = models.FloatField(null=True)
    three_year_z = models.FloatField(null=True)
    one_year_abs_z = models.FloatField(null=True)
    three_year_abs_z = models.FloatField(null=True)

    @classmethod
    def process_net_long(cls, data):
        net_long = (data['Noncommercial Long'] - data['Noncommercial Short'])
        net_long_ratio = net_long/(data['Noncommercial Long'] + data['Noncommercial Short']).diff()
        net_long_ratio.sort_index(inplace=True)
        net_long.sort_index(inplace=True)

        # based on the difference in the net long ratio w/w
        one_year_zscore = (net_long_ratio - net_long_ratio.rolling(1*52).mean()) / net_long_ratio.rolling(1*52).std()
        three_year_zscore = (net_long_ratio - net_long_ratio.rolling(3*52).mean()) / net_long_ratio.rolling(3*52).std()

        # based on the simple absolute number of net long positions
        one_year_abs_zscore = (net_long - net_long.rolling(1*52).mean()) / net_long.rolling(1*52).std()
        three_year_abs_zscore = (net_long - net_long.rolling(3*52).mean()) / net_long.rolling(3*52).std()
        
        latest_date = one_year_zscore.index.max()

        latest_net_long = net_long.loc[net_long.index == latest_date][0]
        latest_one_year_zscore = one_year_zscore.loc[one_year_zscore.index == latest_date][0]
        latest_three_year_zscore = three_year_zscore.loc[three_year_zscore.index == latest_date][0]
        latest_one_year_abs_zscore = one_year_abs_zscore.loc[three_year_zscore.index == latest_date][0]
        latest_three_year_abs_zscore = three_year_abs_zscore.loc[three_year_zscore.index == latest_date][0]

        return latest_net_long, latest_one_year_zscore, latest_three_year_zscore, latest_one_year_abs_zscore, latest_three_year_abs_zscore
    
    @classmethod
    def update(cls):
        quandl_codes = {
            'Gold': '088691',
            'Feeder Cattle': '061641',
            'Canadian Dollar': '090741',
            'Live Cattle': '057642',
            'VIX Futures': '1170E1',
            'Copper': '085692',
            'US Treasury Bond': '020601',
            'Crude Oil': '067651',
            '10 Year Note': '043602',
            '2 Year Note': '042601',
            '5 Year Note': '044601',
            'S&P 500': '13874P',
            'Russell 2000': '239742',
            'Nasdaq': '20974P',
            'United States Dollar': '098662',
            'DJIA': '12460P',
            'Lean Hogs': '054642',
            'Coffee': '083731',
            'Japanese Yen': '097741',
            'Bitcoin': '133741',
            'Corn': '002602',
            'Wheat SRW': '001602',
            'Wheat HRW': '001612',
            'Soybeans': '005602',
            'Random Length Lumber': '058643',
        }
        sub_code = '_FO_L_ALL'

        quandl.ApiConfig.api_key = settings.QUANDL_KEY

        for key in quandl_codes:
            mydata = quandl.get(f"CFTC/{quandl_codes[key]}{sub_code}")
            net_long, one_year, three_year, one_year_abs, three_year_abs = cls.process_net_long(mydata)  
            obj, created = cls.objects.update_or_create(
                date=mydata.index.max(), symbol=key, defaults={
                    'one_year_z': one_year, 
                    'three_year_z': three_year,
                    'one_year_abs_z': one_year_abs,
                    'three_year_abs_z': three_year_abs,
                    'net_long': net_long
                }
            )

    class Meta:
        unique_together = [['symbol', 'date']]


class SignalTimeSeries(models.Model):
    id = models.BigIntegerField(primary_key=True)
    run_time = models.DateTimeField(null=False)
    analysis_label = models.TextField(null=False)
    ticker = models.TextField(null=True)
    signal = models.FloatField(null=True)
    target_date = models.DateField(null=False)

    class Meta:
        db_table = 'signal_time_series'
        managed = False
