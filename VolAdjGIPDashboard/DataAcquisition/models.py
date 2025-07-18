import datetime
import logging
import io
import itertools
import math
import requests

from django.db import models
from django.db.models import F, Max, Q
from django.db.utils import IntegrityError
from django.conf import settings

from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
from tqdm import tqdm
import yfinance
import quandl

import pandas as pd
import numpy as np

QUAD_RETURN_BATCH_SIZE = 5000

class QuadReturn(models.Model):
    quarter_end_date = models.DateField()
    data_start_date = models.DateField()
    data_end_date = models.DateField()
    label = models.CharField(max_length=100)

    prices_updated = models.DateTimeField() # data taken from YahooHistory.updated

    quad_return = models.FloatField()
    quad_stdev = models.FloatField()

    # linear model output
    linear_eoq_forecast = models.FloatField(null=True)
    linear_eoq_r2 = models.FloatField(null=True)
    linear_eoq_95pct = models.FloatField(null=True)

    def __str__(self):
        return f"{self.label} between {self.data_start_date} and {self.data_end_date} (Q={self.quarter_end_date}) returned {100*self.quad_return:.2f}% with a stdev of {100*self.quad_stdev:.2f}% "
    
    @classmethod
    def clear_models(cls):
        cls.objects.update(linear_eoq_forecast=None, linear_eoq_r2=None, linear_eoq_95pct=None)
    
    @classmethod
    def update_models(cls):
        updated_items = list()

        for item in cls.objects.filter(linear_eoq_95pct__isnull=True):
            quad_returns = cls.objects\
                .filter(quarter_end_date=item.quarter_end_date, label=item.label, quad_stdev__gt=0, data_end_date__lte=item.data_end_date)\
                .order_by('data_end_date')\
                .annotate(score=F('quad_return')/F('quad_stdev'))
            
            prior_quad_end_date = (item.quarter_end_date - pd.tseries.offsets.QuarterEnd(n=1)).date()
            current_quad_start = prior_quad_end_date + datetime.timedelta(days=1)            

            day_index = [(qtrn.data_end_date-current_quad_start).days for qtrn in quad_returns]
            
            # you need two points to make a line
            if len(day_index) < 2:
                continue
            
            X = np.array(day_index).reshape(-1, 1)
            y = np.array([qrtn.score for qrtn in quad_returns]).reshape(-1, 1)

            # Regression of performance
            reg = LinearRegression(fit_intercept=False).fit(X=X, y=y)
            
            # the final value only
            item.linear_eoq_forecast = reg.coef_.item()*90.0            
            item.linear_eoq_r2 = reg.score(X, y)

            residuals = [ abs(x.score - reg.coef_.item()*day_index[i]) for i, x in enumerate(quad_returns) ]
            item.linear_eoq_95pct = np.percentile(residuals, 95)   

            updated_items.append(item)

            # manually batching to reduce memory usage
            if len(updated_items) >= QUAD_RETURN_BATCH_SIZE:
                cls.objects.bulk_update(updated_items, ['linear_eoq_forecast', 'linear_eoq_r2', 'linear_eoq_95pct'])
                updated_items = list()

        if updated_items:
            cls.objects.bulk_update(updated_items, ['linear_eoq_forecast', 'linear_eoq_r2', 'linear_eoq_95pct'])

    @classmethod
    def update(cls):
        cls.update_models()

    class Meta:
        unique_together = [['quarter_end_date', 'data_start_date', 'data_end_date', 'label']]


class SecurityHistory(models.Model):
    date = models.DateField()
    ticker = models.CharField(max_length=36)
    close_price = models.FloatField()
    updated = models.DateTimeField(auto_now=True)
    realized_volatility = models.FloatField(null=True) 

    # this is very slow.
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

        distinct_dates = list(history.values('date').distinct().values_list('date', flat=True))
        
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

        # calculate weighted-average rolling volatility
        temp_sum = 0.0
        temp_divider = 0.0

        for leg, quantity in prior_positioning.items():
            vol = history.get(ticker=leg, date=distinct_dates[-1]).realized_volatility

            if vol is None:
                temp_sum = None
                break
            else:
                temp_sum += vol * quantity
                temp_divider += quantity
        
        quad_stdev = temp_sum/temp_divider if temp_sum is not None else np.inf       
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
        return [ [x.upper()] for x in cls.objects.values_list('ticker', flat=True).distinct() ]

    @classmethod
    def update_quad_return(cls, first_date=None, ticker=None, tickers=None, full_run=False):
        logger = logging.getLogger('SecurityHistory.update_quad_return')
        
        if ticker is None and tickers is None:
            tickers = cls.core_tickers()
        elif ticker is not None:
            tickers = [[ticker,]]

        for labels in tqdm(tickers):
            sortable = labels
            sortable.sort()
            modified_label =  cls.__name__ + "_" + ','.join(sortable)

            latest_date = cls.objects.filter(ticker__in=labels).latest('date').date
            first_date = first_date if first_date is not None else cls.objects.filter(ticker__in=labels).earliest('date').date

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
                logger.debug("Tickers=%s, date=%s ... ", labels, try_date)
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
    def equal_volatility_position(cls, tickers: list, lookback=28, target_value=10000, max_date=None) -> dict:
        """ returns the target position count, per ticker, for a given target value.

        Args:
            tickers (_type_): _description_
            lookback (int, optional): _description_. Defaults to 28.
            target_value (int, optional): _description_. Defaults to 10000.
            max_date (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            dict: {ticker: position_size, ...}
        """        
        logger = logging.getLogger('SecurityHistory.equal_volatility_position')
        logger.debug("function triggered for tickers=[%s], lookback=%s, target_value=%s, max_date=%s", tickers, lookback, target_value, max_date)

        standard_move = dict()
        last_price_lookup = dict()

        controlling_leg = None
        max_price = None

        dataframe = cls.dataframe(max_date=max_date, tickers=tickers, lookback=lookback)
        if len(tickers) == 1:
            subset = dataframe[dataframe.index.get_level_values('ticker') == tickers[0]].dropna()
            
            return {
                tickers[0]: math.floor(target_value / subset.iloc[-1].close_price)
            }

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
        
        # here is where realized_vol gets calculated
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
        return f"{self.ticker} on {self.date} was ${self.close_price:.2f}"

    class Meta:
        abstract = True


class YahooHistory(SecurityHistory):
    @classmethod
    def core_tickers(cls):
        tickers = [ [x.upper()] for x in cls.objects.values_list('ticker', flat=True).distinct() ]
        tickers.append(['QQQ', 'XLF', 'XLI'])
        tickers.append(['TLT', 'UUP', 'VPU'])
        tickers.append(['GLD', 'VPU'])
        
        for ticker in ('CAD=X', 'SPY', 'DJP', 'XLE', 'GLD', 'VTI', 'VEU'):
            if [ticker] not in tickers:
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
        return f"{self.ticker} on {self.date} was ${self.close_price:.2f} with 1-week vol {100*self.realized_volatility:.2f}% "

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

QUAD_FORECAST_BATCH_SIZE = 5000

class QuadForecasts(models.Model):
    quarter_end_date = models.DateField()
    date = models.DateField()

    cpi_roc = models.FloatField()
    gdp_roc = models.FloatField()
    quad = models.IntegerField()

    updated = models.DateTimeField(auto_now=True)

    # model outputs
    cpi_origin = models.FloatField(null=True) # i.e: mean
    cpi_sigma = models.FloatField(null=True) # all time stdev
    
    gdp_origin = models.FloatField(null=True) # i.e.: mean
    gdp_sigma = models.FloatField(null=True) # all time stdev
    
    @classmethod
    def clear_models(cls):
        cls.objects.update(cpi_origin=None, gdp_origin=None, cpi_sigma=None, gdp_sigma=None)
    
    @classmethod
    def update_models(cls):
        # establish prior
        all_items = cls.objects.values_list('cpi_roc', 'gdp_roc')

        empty_objects = cls.objects.filter(
            Q(cpi_origin__isnull=True) | Q(gdp_origin__isnull=True) |
            Q(cpi_sigma__isnull=True) | Q(gdp_sigma__isnull=True)
        )

        std_cache = dict()
        
        updated_items = list()
        for item in empty_objects:
            data_series = cls.objects.filter(quarter_end_date=item.quarter_end_date, date__lte=item.date).values_list('gdp_roc', 'cpi_roc')
            gdp_mean, cpi_mean = np.mean(data_series, axis=0)

            if item.date in std_cache:
                gdp_sigma, cpi_sigma = std_cache[item.date]
            else:
                sub_items = all_items.filter(date__lte=item.date)
                gdp_sigma, cpi_sigma = np.std(sub_items, axis=0)
                std_cache[item.date] = (gdp_sigma, cpi_sigma)

            # update parameters
            item.cpi_origin = cpi_mean
            item.gdp_origin = gdp_mean
            item.cpi_sigma = cpi_sigma
            item.gdp_sigma = gdp_sigma

            updated_items.append(item)

            if len(updated_items) >= QUAD_FORECAST_BATCH_SIZE:
                cls.objects.bulk_update(updated_items, ['cpi_origin', 'gdp_origin', 'cpi_sigma', 'gdp_sigma'])
                updated_items = list()
        
        if updated_items:
            cls.objects.bulk_update(updated_items, ['cpi_origin', 'gdp_origin', 'cpi_sigma', 'gdp_sigma'])

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

        usa_quads = usa_quads[
            (usa_quads.index.get_level_values('date') <= usa_quads.index.get_level_values('quarter_end_date'))
        ]

        for row in usa_quads.itertuples():
            quarter, date = row.Index

            try:
                _quad = int(row.quad)
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
        
        cls.update_models()

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
