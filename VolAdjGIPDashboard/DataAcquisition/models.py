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

class QuadReturn(models.Model):
    quarter_end_date = models.DateField()
    data_start_date = models.DateField()
    data_end_date = models.DateField()
    label = models.CharField(max_length=100)

    prices_updated = models.DateTimeField() # data taken from YahooHistory.updated

    quad_return = models.FloatField()

    class Meta:
        unique_together = [['quarter_end_date', 'data_start_date', 'data_end_date', 'label']]
    

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
            results = results.filter(date__gte=max_date - datetime.timedelta(days=lookback*2)) # this math is impercise because of weekends
        
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
        logger.debug(f"function triggered for tickers=[{tickers}], lookback={lookback}, target_value={target_value}, max_date={max_date}")

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
            logger.debug(f"{security} close={latest_close}, realized_vol={realized_vol}")
            
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
    def quad_return(cls, tickers, date_within_quad):
        tickers.sort() # make the list deterministic for the same input (used for label later)

        current_quad = QuadForecasts.objects.filter(date__lte=date_within_quad).latest('quarter_end_date', 'date')
        print(f"current_quad quarter={current_quad.quarter_end_date} date={current_quad.date}")

        # this is the last known date for the prior quad
        start_date = QuadForecasts.objects.filter(date__lt=current_quad.date).exclude(quad=current_quad.quad, quarter_end_date__gte=current_quad.quarter_end_date).latest('quarter_end_date', 'date').quarter_end_date
        print(f"last known date for prior quad: {start_date}")
        
        # this is when we started this quad
        start_date = QuadForecasts.objects.filter(date__gt=start_date, quad=current_quad.quad).earliest('date').date
        
        history = cls.objects.filter(ticker__in=tickers, date__gte=start_date, date__lte=date_within_quad).order_by('date')

        try:
            cached = QuadReturn.objects.filter(
                quarter_end_date = current_quad.quarter_end_date,
                data_start_date = start_date,
                data_end_date = date_within_quad, 
                label = ','.join(tickers).upper(),
                prices_updated__gte = history.latest('updated').updated
            ).latest('prices_updated')

            return cached.quad_return

        except QuadReturn.DoesNotExist:
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

        quad_return = end_market_value / start_market_value - 1
        
        QuadReturn.objects.filter(
            quarter_end_date = current_quad.quarter_end_date,
            data_start_date = start_date, 
            data_end_date = date_within_quad, 
            label = ','.join(tickers).upper(),
        ).delete() # if we had an older one, kill it
        
        cached_return = QuadReturn(
            quarter_end_date = current_quad.quarter_end_date, 
            data_end_date = date_within_quad,
            data_start_date = start_date,
            label = ','.join(tickers).upper(),
            prices_updated = history.latest('updated').updated,
            quad_return = quad_return 
        )
        cached_return.save()
        
        return end_market_value / start_market_value - 1

    class Meta:
        unique_together = [['ticker', 'date']]


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

        for quarter, cpi in cpi_forecasts.iteritems():
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
    def fetch_usa_gi_data(cls):
        '''
        Fetches the latest GDP and CPI numbers + forecasts. Excludes older forecast data.
        '''

        start_date = datetime.date(2008, 1, 1)

        # Real GDP, seasonally adjusted. Quarterly.
        gdp_data = web.DataReader('GDPC1', 'fred', start = start_date)['GDPC1']

        # align the FRED quarterly dates to Pandas quarterly dates
        # each index value will be the last day of a quarter. i.e. 2019-06-30 is Q2 2019.
        gdp_data.index = gdp_data.index.shift(1, freq='Q')   
        gdp_data = gdp_data.resample('Q').asfreq()

        # CPI, all items, urban, not seasonally adjusted. Monthly.
        cpi_all_urban_unadjusted_data = web.DataReader('CPIAUCNS', 'fred', start = start_date)['CPIAUCNS']    
        cpi_data = cpi_all_urban_unadjusted_data.resample('Q').mean()

        cpi_nowcasts = CPIForecast.dataframe()        
        latest_date = cpi_nowcasts.date.max()

        cpi_data = pd.concat([cpi_data[~(cpi_data.index.isin(cpi_nowcasts.cpi.index))], cpi_nowcasts.cpi])

        return gdp_data, cpi_data, latest_date

    @classmethod
    def get_new_york_fed_gdp_nowcasts(cls):
        url = r'https://www.newyorkfed.org/medialibrary/media/research/policy/nowcast/new-york-fed-staff-nowcast_data_2002-present.xlsx?la=en'

        response = requests.get(url, allow_redirects=True)
        memory_handle = io.BytesIO(response.content)

        data = pd.read_excel(memory_handle, sheet_name='Forecasts By Quarter', header=13)
        latest_date = data['Forecast Date'].max().date()

        data = data.melt(id_vars=['Forecast Date'], var_name="quarter", value_name="forecast")
        data.columns = ['date', 'quarter', 'forecast']
        data['quarter'] = pd.to_datetime(data.quarter) + pd.offsets.QuarterEnd()*0
        
        return data, latest_date

    @classmethod
    def get_gdp_set(cls, actual_gdp):
        data, latest_date = cls.get_new_york_fed_gdp_nowcasts()

        data.set_index(['quarter', 'date'], inplace=True)

        data['growth'] = (data.forecast/100 + 1)**(1/4)
        data.drop(['forecast'], inplace=True, axis='columns')
        data.dropna(inplace=True)

        gdp_df = pd.DataFrame({
            'date': actual_gdp.index + pd.offsets.QuarterEnd(), # pre-shift, for easy multiplying later
            'prior_actual_gdp': actual_gdp.values
        }).set_index('date')

        first_order_estimates = data.join(gdp_df, on='quarter')
        first_order_estimates['number'] = (first_order_estimates.growth * first_order_estimates.prior_actual_gdp)
        first_order_estimates.drop(['prior_actual_gdp'], inplace=True, axis='columns')

        forecasted_gdp = first_order_estimates.reset_index()
        forecasted_gdp.quarter += pd.offsets.QuarterEnd() # shift for easy multiplying later
        forecasted_gdp.drop(['growth'], axis='columns', inplace=True)
        forecasted_gdp = forecasted_gdp.rename({'number': 'gdp'}, axis='columns').dropna().set_index(['quarter', 'date'])

        actual_current_gdp = pd.DataFrame({
            'date': actual_gdp.index,
            'actual_gdp': actual_gdp.values
        }).set_index('date')

        second_order_estimates = pd.concat([forecasted_gdp, first_order_estimates], join='outer', sort=True, axis=1)
        second_order_estimates = second_order_estimates.join(actual_current_gdp, on='quarter')

        second_order_estimates = second_order_estimates.assign(
            best_estimate = np.where(
                second_order_estimates.actual_gdp.isnull(),
                np.where(
                    second_order_estimates.number.isnull(), # original gdp forecast blank
                    second_order_estimates.gdp * first_order_estimates.growth,  # use forecast applied to a forecast (second order)
                    second_order_estimates.number # otherwise use original gdp forecast (first order)
                ),
                second_order_estimates.actual_gdp
            )
        )

        second_order_estimates.drop(['gdp', 'number', 'actual_gdp'], inplace=True, axis='columns') # these columns are confusing anyway due to shifting
        second_order_estimates.dropna(how='all', inplace=True) # if all columns are null, drop
        
        return second_order_estimates, latest_date

    @classmethod
    def determine_quads(cls, actual_gdp, actual_cpi):
        dataframe, latest_date = cls.get_gdp_set(actual_gdp)
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
        merged_data.index.names = ['quarter']
        merged_data.index += pd.offsets.QuarterEnd() # push forward one quarter

        dataframe = dataframe.join(merged_data, on='quarter')

        # do maths
        dataframe['gdp_yoy'] = dataframe.best_estimate / dataframe.past_gdp - 1
        dataframe['cpi_yoy'] = dataframe.cpi / dataframe.past_cpi - 1

        # quarterly rate of change of the yoy rates
        dataframe['gdp_roc'] = (dataframe.gdp_yoy - dataframe.past_gdp_yoy) * 1e4 # bps
        dataframe['cpi_roc'] = (dataframe.cpi_yoy - dataframe.past_cpi_yoy) * 1e4 # bps

        dataframe['quad'] = dataframe.groupby(['quarter', 'date']).apply(cls.__determine_quad_multi_index).rename('quad')

        # drop confusing intermediates
        dataframe.drop([
            'cpi', 'best_estimate', 'past_gdp', 'past_cpi', 'gdp_yoy', 'cpi_yoy',
            'past_cpi_yoy', 'past_gdp_yoy',
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
            (usa_quads.index.get_level_values('date') <= usa_quads.index.get_level_values('quarter')) &
            #(usa_quads.index.get_level_values('date') > usa_quads.index.get_level_values('quarter') - pd.offsets.QuarterEnd()) &
            (usa_quads.index.get_level_values('date') > max_date - datetime.timedelta(days=365))
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
