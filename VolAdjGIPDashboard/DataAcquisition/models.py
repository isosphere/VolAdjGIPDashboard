import datetime
import logging
import io
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
    def equal_volatility_position(cls, tickers, lookback=28, target_value=10000, max_date=None):
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
    def fetch_usa_cpi_nowcasts(cls):
        url = r'https://www.forecasts.org/inf/cpi-data.csv'
    
        response = requests.get(url, allow_redirects=True)
        memory_handle = io.BytesIO(response.content)
        
        dataframe = pd.read_csv(memory_handle, index_col="Date", header=0, names=["Date", "CPI", "Note"], skipfooter=2, engine="python")
        dataframe.index= pd.to_datetime(dataframe.index)
        cpi_forecasts = dataframe.loc[dataframe.Note.str.contains("Forecast")]["CPI"].resample('Q').mean()
        
        return cpi_forecasts


    @classmethod
    def fetch_usa_gdp_nowcasts(cls):        
        start_date = datetime.date(2001,1,1)
        gdp_nowcasts = web.DataReader('GDPNOW', 'fred', start = start_date)['GDPNOW'] # Real GDP, seasonally adjusted. Quarterly. 

        # align the FRED quarterly dates to Pandas quarterly dates
        gdp_nowcasts.index = gdp_nowcasts.index.shift(1, freq='Q')
        gdp_nowcasts = gdp_nowcasts.resample('Q').asfreq()

        return gdp_nowcasts

    @classmethod
    def fetch_usa_gi_data(cls):
        start_date = datetime.date(2001,1,1)
        # Real GDP, seasonally adjusted. Quarterly.
        gdp_data = web.DataReader('GDPC1', 'fred', start = start_date)['GDPC1']

        # align the FRED quarterly dates to Pandas quarterly dates
        # each index value will be the last day of a quarter. i.e. 2019-06-30 is Q2 2019.
        gdp_data.index = gdp_data.index.shift(1, freq='Q')

        gdp_nowcasts = cls.fetch_usa_gdp_nowcasts()
        future_nowcasts = 1 + gdp_nowcasts[gdp_nowcasts.index > gdp_data.index.max()] / 100

        # shift and multiply
        shifted_gdp = gdp_data.copy()
        shifted_gdp.index = gdp_data.index.shift(4, freq='Q')

        future_nowcasts = future_nowcasts.multiply(shifted_gdp).dropna()

        gdp_data = pd.concat([gdp_data, future_nowcasts])        

        gdp_data = gdp_data.resample('Q').asfreq()

        # CPI, all items, urban, not seasonally adjusted. Monthly.
        cpi_all_urban_unadjusted_data = web.DataReader('CPIAUCNS', 'fred', start = start_date)['CPIAUCNS']    
        cpi_data = cpi_all_urban_unadjusted_data.resample('Q').mean()

        cpi_nowcasts = cls.fetch_usa_cpi_nowcasts()
        cpi_data = pd.concat([cpi_data, cpi_nowcasts])

        return gdp_data, cpi_data            

    @classmethod
    def get_new_york_fed_gdp_nowcasts(cls):
        url = r'https://www.newyorkfed.org/medialibrary/media/research/policy/nowcast/new-york-fed-staff-nowcast_data_2002-present.xlsx?la=en'

        response = requests.get(url, allow_redirects=True)
        memory_handle = io.BytesIO(response.content)

        data = pd.read_excel(memory_handle, sheet_name='Forecasts By Horizon', header=13).iloc[:, :5]
        data.columns = ['date', 'quarter', 'backcast', 'nowcast', 'forecast']
        data['quarter'] = pd.to_datetime(data.quarter) + pd.offsets.QuarterEnd()

        return data

    @classmethod
    def get_gdp_set(cls, actual_gdp):
        fed_nowcasts = cls.get_new_york_fed_gdp_nowcasts()

        forecasts = fed_nowcasts.copy()
        forecasts.quarter += pd.offsets.QuarterEnd()
        forecasts.drop(['backcast', 'nowcast'], inplace=True, axis='columns')
        forecasts.set_index(['quarter', 'date'], inplace=True)

        nowcasts = fed_nowcasts.copy()
        nowcasts.drop(['backcast', 'forecast'], inplace=True, axis='columns')
        nowcasts.set_index(['quarter', 'date'], inplace=True)

        data = (pd.concat([forecasts, nowcasts], join='outer', sort=True, axis=1))

        data = data.assign(growth = np.where(data.nowcast.isnull(), data.forecast, data.nowcast))
        data.growth = (data.growth/100 + 1)**(1/4)
        data.drop(['forecast', 'nowcast'], inplace=True, axis='columns')
        data.dropna(inplace=True)

        gdp_df = pd.DataFrame({
            'date': actual_gdp.index + pd.offsets.QuarterEnd(), # pre-shift, for easy multiplying later
            'gdp': actual_gdp.values
        }).set_index('date')

        first_order_estimates = data.join(gdp_df, on='quarter')
        first_order_estimates['number'] = (first_order_estimates.growth * first_order_estimates.gdp)
        first_order_estimates.drop(['gdp'], inplace=True, axis='columns')

        forecasted_gdp = first_order_estimates.reset_index()
        forecasted_gdp.quarter += pd.offsets.QuarterEnd() # shift for easy multiplying later
        forecasted_gdp.drop(['growth'], axis='columns', inplace=True)
        forecasted_gdp = forecasted_gdp.rename({'number': 'gdp'}, axis='columns').dropna().set_index(['quarter', 'date'])

        second_order_estimates = pd.concat([forecasted_gdp, first_order_estimates], join='outer', sort=True, axis=1)
        second_order_estimates = second_order_estimates.assign(
            best_estimate = np.where(
                second_order_estimates.number.isnull(), # original gdp forecast blank
                second_order_estimates.gdp * first_order_estimates.growth,  # use forecast applied to a forecast (second order)
                second_order_estimates.number # otherwise use original gdp forecast (first order)
            )
        )

        second_order_estimates.drop(['gdp', 'number'], inplace=True, axis='columns') # these columns are confusing anyway due to shifting
        second_order_estimates.dropna(how='all', inplace=True) # if all columns are null, drop
        
        return second_order_estimates

    @classmethod
    def determine_quads(cls, actual_gdp, actual_cpi):
        dataframe = cls.get_gdp_set(actual_gdp)
        dataframe.drop('growth', inplace=True, axis='columns')

        # Collect required GDP numbers
        shifted_gdp = pd.DataFrame({'quarter': actual_gdp.index + 4*pd.offsets.QuarterEnd(), 'past_gdp': actual_gdp}) # final numbers
        shifted_gdp.set_index('quarter', inplace=True)
        dataframe = dataframe.join(shifted_gdp, on='quarter')

        # Collect required CPI numbers
        cpi_df = pd.DataFrame({'quarter':actual_cpi.index, 'cpi':actual_cpi.values})
        cpi_df.set_index('quarter', inplace=True)
        dataframe = dataframe.join(cpi_df, on='quarter')

        shifted_cpi = pd.DataFrame({'quarter': actual_cpi.index + 4*pd.offsets.QuarterEnd(), 'past_cpi': actual_cpi}) # final numbers
        shifted_cpi.set_index('quarter', inplace=True)
        dataframe = dataframe.join(shifted_cpi, on='quarter')

        # Collected settled YOY numbers
        merged_data = pd.DataFrame({
            'past_cpi_yoy': actual_cpi,
            'past_gdp_yoy': actual_gdp,
        }).pct_change(4)
        merged_data.index.names=['quarter']
        merged_data.index += pd.offsets.QuarterEnd()

        dataframe = dataframe.join(merged_data, on='quarter')

        # do maths
        dataframe['gdp_yoy'] = dataframe.best_estimate / dataframe.past_gdp - 1
        dataframe['cpi_yoy'] = dataframe.cpi / dataframe.past_cpi - 1

        dataframe['gdp_roc'] = (dataframe.gdp_yoy - dataframe.past_gdp_yoy) * 1e4 # bps
        dataframe['cpi_roc'] = (dataframe.cpi_yoy - dataframe.past_cpi_yoy) * 1e4 # bps

        dataframe['quad'] = dataframe.groupby(['quarter', 'date']).apply(cls.__determine_quad_multi_index).rename('quad')

        # drop confusing intermediates
        dataframe.drop([
            'cpi', 'best_estimate', 'past_gdp', 'past_cpi', 'gdp_yoy', 'cpi_yoy',
            'past_cpi_yoy', 'past_gdp_yoy',
        ], inplace=True, axis='columns')    

        return dataframe

    @classmethod
    def update(cls):
        gdp, cpi = cls.fetch_usa_gi_data()
        usa_quads = cls.determine_quads(gdp, cpi)

        for row in usa_quads.itertuples():
            quarter, date = row.Index

            try:
                quad = int(row.quad)
            except ValueError:
                continue

            cls.objects.get_or_create(
                quarter_end_date = quarter.date(), 
                date = date.date(), 
                defaults = {
                    'updated': datetime.datetime.now(),
                    'cpi_roc': row.cpi_roc,
                    'gdp_roc': row.gdp_roc,
                    'quad': int(row.quad)                   
                }
            )

    class Meta:
        unique_together = [['quarter_end_date', 'date']]
