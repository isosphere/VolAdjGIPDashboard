import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from DataAcquisition.models import YahooHistory, QuadForecasts, QuadReturn
from django.db.models import F
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render
from django.http import Http404

#  todo: this should be reusable in the index
def quad_performance(request, label):
    label = label.replace(' ', '')
    latest_date = YahooHistory.objects.latest('date').date
    
    current_quad_forecast = QuadForecasts.objects.filter(quarter_end_date=latest_date + pd.tseries.offsets.QuarterEnd(n=0)).latest('date')
    current_quad, quarter_end_date = current_quad_forecast.quad, current_quad_forecast.quarter_end_date
    prior_quad_end_date = (quarter_end_date - pd.tseries.offsets.QuarterEnd(n=1)).date()
    current_quad_start = prior_quad_end_date + datetime.timedelta(days=1)
    prior_quad_start = (prior_quad_end_date - pd.tseries.offsets.QuarterEnd(n=1) + datetime.timedelta(days=1)).date()

    # time series data for quad return charts
    quad_returns = QuadReturn.objects.filter(quarter_end_date=quarter_end_date, label=label)\
        .order_by('label', 'data_end_date')\
        .annotate(score=F('quad_return')/F('quad_stdev'))
    
    prior_quad_returns = QuadReturn.objects.filter(quarter_end_date=prior_quad_end_date, label=label).exclude(quad_stdev=0)\
        .order_by('label', 'data_end_date')\
        .annotate(score=F('quad_return')/F('quad_stdev'))

    if not quad_returns:
        raise Http404(f"Label {label} does not exist.")

    quad_performance = list()
    for ticker_lookup, date, score in quad_returns.values_list('label', 'data_end_date', 'score'):
        if score:
            quad_performance.append(((date-current_quad_start).days, round(score, 2)))
        else:
            continue

    prior_quad_performance = list()
    for ticker_lookup, date, score in prior_quad_returns.values_list('label', 'data_end_date', 'score'):
        if score:
            prior_quad_performance.append(((date-prior_quad_start).days, round(score, 2)))
        else:
            continue

    current_regression = quad_returns.latest('data_end_date').linear_eoq_forecast
    prior_regression = prior_quad_returns.latest('data_end_date').linear_eoq_forecast
    error_percentile = prior_quad_returns.latest('data_end_date').linear_eoq_95pct

    latest_performance = quad_returns.latest('data_end_date').data_end_date

    current_performance = quad_returns.get(data_end_date=latest_performance).score
    prior_performance = quad_returns.exclude(data_end_date=latest_performance).latest('data_end_date').score
    performance_change = round(current_performance - prior_performance, 1)
    
    return render(request, 'UserInterface/performance.htm', {
        'label': label,
        'performance_change': performance_change,
        'quad_performance': quad_performance,
        'prior_quad_performance': prior_quad_performance,
        'current_regression': current_regression,
        'prior_regression': prior_regression,
        'error_percentile': error_percentile,
    })

def classify_r2(r2: float | None) -> str:
    # expects 100 rather than 1.0, i.e.: base unit is pct

    if not r2:
        return "none"

    r2 = abs(r2)

    if r2 < 30:
        return "none"
    elif r2 < 50:
        return "weak"
    elif r2 < 70:
        return "moderate"
    else:
        return "strong"

def all_symbol_summary(quad_allocation, latest_date):
    """ returns something like:

    [ [ label, close_price, ratio, last_week_vol, linear_r2, linear_eoq_forecast, linear_r2_quality ], ... ]
    linear_r2_quality is in ('none', 'weak', 'moderate', 'strong') 
    
    Args:
        quad_allocation (_type_): _description_
        latest_date (_type_): _description_

    Returns:
        _type_: _description_
    """
    symbol_values = dict()

    for group in (YahooHistory,):
        if group.__name__ == 'YahooHistory':
            all_symbols = list()
            for quad in quad_allocation:
                for symbol in quad_allocation[quad]:
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)

            all_symbols += list(group.objects.all().values_list('ticker', flat=True).distinct())
        else:
            all_symbols = list(group.objects.all().values_list('ticker', flat=True).distinct())

        all_symbols.sort()

        for symbol in all_symbols:
            try:
                symbol_data = group.objects.get(ticker=symbol, date=latest_date)
            except group.DoesNotExist:
                symbol_values[symbol] = [
                    group.__name__ + '_' + symbol,
                    '--.--',
                    '--.--',
                    '--.--',
                    '--',
                    '--.--',
                    'none'
                ]
                continue

            prior_week_ref = group.objects.filter(ticker=symbol).latest('date').date - datetime.timedelta(weeks=1)
            prior_week = prior_week_ref.isocalendar()[1]

            try:
                last_week = group.objects.filter(ticker=symbol, date__week=prior_week).latest('date')
            except group.DoesNotExist:
                symbol_values[symbol] = [
                    group.__name__ + '_' + symbol,
                    '--.--',
                    '--.--',
                    '--.--',
                    '--',
                    '--.--',
                    'none'
                ]
                continue
            
            last_week_date, last_week_val = last_week.date, last_week.close_price

            # get realized vol as of last week - don't let the buy/sell targets move with current vol
            last_week_vol = group.objects.get(ticker=symbol, date=last_week_date).realized_volatility
            
            try:
                current_performance = QuadReturn.objects.filter(label=f"{group.__name__}_{symbol}").latest('quarter_end_date', 'data_end_date')
                if current_performance and current_performance.quad_stdev:
                    current_score = current_performance.quad_return / current_performance.quad_stdev
                    prior_performance = QuadReturn.objects.filter(label=current_performance.label).exclude(quarter_end_date=current_performance.quarter_end_date).latest('data_end_date')
                else:
                    current_performance = None
                    prior_performance = None
            except QuadReturn.DoesNotExist:
                current_performance = None
                current_score = None
                prior_performance = None

            if prior_performance and prior_performance.linear_eoq_r2:
                r_squared = prior_performance.linear_eoq_r2 * 100
            else:
                r_squared = None

            if last_week_vol is not None:
                symbol_values[symbol] = [
                    group.__name__ + '_' + symbol,                    
                    symbol_data.close_price, 
                    '--.--' if not current_score else current_score,
                    100*last_week_vol,    
                    r_squared,
                    current_performance.linear_eoq_forecast if current_performance else '--.--',
                    classify_r2(r_squared)
                ]
            else:
                symbol_values[symbol] = [
                    group.__name__ + '_' + symbol,
                    symbol_data.close_price, 
                    '--.--' if not current_score else current_score,
                    '--.--', 
                    r_squared,
                    current_performance.linear_eoq_forecast if current_performance else '--.--',
                    classify_r2(r_squared)
                ]

    return symbol_values


def quad_performance_summary(quad_allocation, prior_quad_end_date, current_quad_start):
    current_quad_return = dict()
    prior_quad_return = dict()

    for quad in quad_allocation:
        try_date = datetime.date.today()
        attempts = 7

        while True:
            try:
                current_quad_return[quad] = list(YahooHistory.quad_return(
                    tickers=quad_allocation[quad], 
                    date_within_quad=try_date
                ))

                current_quad_return[quad].append(round(
                    current_quad_return[quad][0]/current_quad_return[quad][1],
                ndigits=1))

                current_quad_return[quad][0] = round(current_quad_return[quad][0]*100, ndigits=1)
                current_quad_return[quad][1] = round(current_quad_return[quad][1]*100, ndigits=1)

                prior_quad_return[quad] = list(YahooHistory.quad_return(
                    tickers=quad_allocation[quad], 
                    date_within_quad=prior_quad_end_date
                ))

                prior_quad_return[quad].append(round(prior_quad_return[quad][0]/prior_quad_return[quad][1], ndigits=1))

                prior_quad_return[quad][0] = round(prior_quad_return[quad][0]*100, ndigits=1)
                prior_quad_return[quad][1] = round(prior_quad_return[quad][1]*100, ndigits=1)

                break
        
            except YahooHistory.DoesNotExist:
                try_date -= datetime.timedelta(days=1)
                attempts -= 1
                if attempts == 0:
                    current_quad_return[quad] = "N/A", "N/A"
                    break

            except QuadForecasts.DoesNotExist:
                current_quad_return[quad] = "N/A", "N/A"
                break
    
    return current_quad_return, prior_quad_return


def index(request, default_net_liquidating_value=10000, lookback=52, default_currency='USD'):
    # For our little 4-quad chart
    current_date = datetime.date.today()
    quarter_int = (current_date.month - 1) // 3 + 1 
    quarter_date = datetime.date(current_date.year, 1, 1) + pd.offsets.QuarterEnd(n=1)*quarter_int

    quad_guesses = QuadForecasts.objects.filter(quarter_end_date=quarter_date).order_by('-date')
    current_quad_guess = quad_guesses.latest('date').quad

    latest_gdp_sigma, latest_cpi_sigma = quad_guesses[0].gdp_sigma, quad_guesses[0].cpi_sigma
    latest_gdp_origin, latest_cpi_origin = quad_guesses[0].gdp_origin, quad_guesses[0].cpi_origin

    quad_guesses = quad_guesses[:3].values_list('date', 'gdp_roc', 'cpi_roc')

    # Position sizing inputs
    net_liquidating_value = request.POST.get('value', default_net_liquidating_value)
    currency = request.POST.get('currency', default_currency)

    if currency not in ('USD', 'CAD'):
        currency = default_currency
        messages.error(request, f"The currency you specified is not supported, so you get {default_currency} instead.")
    
    try:
        net_liquidating_value = int(net_liquidating_value)
    except ValueError:
        net_liquidating_value = default_net_liquidating_value
        messages.error(request, f"The net liquidating value you specified was invalid, so you get {default_net_liquidating_value} instead.")
    
    # Price and Standard Move Table
    latest_rate = YahooHistory.objects.filter(ticker='CAD=X').latest('date').close_price
    if currency == 'CAD':
        net_liquidating_value /= latest_rate

    quad_allocation = {
        1: ['QQQ',],
        2: ['XLF', 'XLI', 'QQQ'],
        3: ['GLD','VPU'],
        4: ['VPU', 'TLT', 'UUP'],
#        'Market': ['VTI',]
    }

    weekly_return = dict()
    for quad in quad_allocation:
        current_weekly_return = YahooHistory.weekly_return(quad_allocation[quad])
        weekly_return[quad] = current_weekly_return*100 if current_weekly_return is not None else None

    latest_date = YahooHistory.objects.latest('date').date

    symbol_values = all_symbol_summary(quad_allocation, latest_date)

    # for positioning, at the bottom of our page
    quad_allocations = dict()
    for quad in quad_allocation:
        quad_allocations[quad] = YahooHistory.equal_volatility_position(quad_allocation[quad], target_value=net_liquidating_value)

    # Quad Return Calculation
    try:
        current_quad_forecast = QuadForecasts.objects.filter(quarter_end_date=latest_date + pd.tseries.offsets.QuarterEnd(n=0)).latest('date')
    except QuadForecasts.DoesNotExist:
        current_quad_forecast = QuadForecasts.objects.filter(quarter_end_date=latest_date - pd.tseries.offsets.QuarterEnd(n=1)).latest('date')
    current_quad, quarter_end_date = current_quad_forecast.quad, current_quad_forecast.quarter_end_date
    
    data_updated = YahooHistory.objects.latest('updated').updated
    prior_quad_end_date = (quarter_end_date - pd.tseries.offsets.QuarterEnd(n=1)).date()
    prior_quad_start = (prior_quad_end_date - pd.tseries.offsets.QuarterEnd(n=1) + datetime.timedelta(days=1)).date()
    current_quad_start = prior_quad_end_date + datetime.timedelta(days=1)

    current_quad_return, prior_quad_return = quad_performance_summary(quad_allocation, prior_quad_end_date, current_quad_start)

    try:
        prior_quad = QuadReturn.objects.filter(quarter_end_date=prior_quad_end_date).latest('quarter_end_date', 'data_end_date')
        prior_quad_end = current_quad_start - datetime.timedelta(days=1)
    except QuadReturn.DoesNotExist:
        prior_quad_start, prior_quad_end = '?', '?'

    prior_quad_guess = QuadForecasts.objects.filter(quarter_end_date=prior_quad_end_date).latest('date').quad

    net_liquidating_value = round(net_liquidating_value, 0)

    # time series data for quad return charts
    quad_labels = ('YahooHistory_QQQ', 'YahooHistory_QQQ,XLF,XLI', 'YahooHistory_GLD,VPU', 'YahooHistory_TLT,UUP,VPU')
    quad_returns = QuadReturn.objects.filter(quarter_end_date=quarter_end_date, label__in=quad_labels).order_by('label', 'data_end_date').annotate(score=F('quad_return')/F('quad_stdev'))
    prior_quad_returns = QuadReturn.objects.filter(quarter_end_date=prior_quad_end_date, label__in=quad_labels).order_by('label', 'data_end_date').annotate(score=F('quad_return')/F('quad_stdev'))

    quad_ticker_lookup = dict()
    for quad in quad_allocation:
        tickers = quad_allocation[quad]
        tickers.sort()
        expected_label = "YahooHistory_" + ','.join(tickers).upper()

        quad_ticker_lookup[expected_label] = quad
    
    quad_performance = dict()

    for ticker_lookup, date, score in quad_returns.values_list('label', 'data_end_date', 'score'):
        try:
            quad = quad_ticker_lookup[ticker_lookup]
        except KeyError:
            quad = None

        current_day = (date-current_quad_start).days
        
        if quad is not None:
            if quad not in quad_performance:
                quad_performance[quad] = list()
            
            quad_performance[quad].append((current_day, round(score, 2)))

    prior_quad_performance = dict()
    for ticker_lookup, date, score in prior_quad_returns.values_list('label', 'data_end_date', 'score'):
        try:
            quad = quad_ticker_lookup[ticker_lookup]
        except KeyError:
            continue

        if quad not in prior_quad_performance:
            prior_quad_performance[quad] = list()
        
        prior_quad_performance[quad].append(((date-prior_quad_start).days, round(score, 2)))

    # Regression of performance
    current_regressions = dict()
    prior_regressions = dict()
    prior_error_percentiles = dict()

    for quad in quad_allocation:
        reg = LinearRegression(fit_intercept=False).fit(
            X=np.array(list( map(lambda x: x[0], quad_performance[quad]) )).reshape(-1, 1),
            y=np.array(list( map(lambda x: x[1], quad_performance[quad]) )).reshape(-1, 1)
        )

        # final values
        current_regressions[quad] = reg.coef_.item()*90.0
        
        reg = LinearRegression(fit_intercept=False).fit(
            X=np.array(list( map(lambda x: x[0], prior_quad_performance[quad]) )).reshape(-1, 1),
            y=np.array(list( map(lambda x: x[1], prior_quad_performance[quad]) )).reshape(-1, 1)
        )
        prior_regressions[quad] = reg.coef_.item()*90.0

        prior_error_percentiles[quad] = np.percentile([ abs(x[1] - reg.coef_.item()*x[0]) for x in prior_quad_performance[quad] ], 95)

    performance_change = dict()
    year, prior_weeknum, _ = (latest_date - datetime.timedelta(days=7)).isocalendar()

    for lookup in quad_ticker_lookup:
        quad = quad_ticker_lookup[lookup]

        try:
            current_performance = quad_returns.get(label=lookup, data_end_date=latest_date).score

            try:
                prior_performance = quad_returns.filter(data_end_date__week=prior_weeknum, data_end_date__year=year, label=lookup).latest('data_end_date').score
                performance_change[quad] = round(current_performance - prior_performance, ndigits=1)
            except QuadReturn.DoesNotExist:
                performance_change[quad]  = '--.-'
        except QuadReturn.DoesNotExist:
            performance_change[quad]  = '--.-'
    
    # signal_data = SignalTimeSeries.objects.filter(
    #     target_date__day=quarter_end_date.day,
    #     target_date__month=quarter_end_date.month,
    #     target_date__year=quarter_end_date.year,
    #     analysis_label__contains='market_continuous_multinormal_functional.r .posterior_bullishness'
    # ).order_by('ticker', 'run_time')

    # selected_tickers = {'QQQ', 'XLF', 'XLI', 'SPY', 'GLD', 'XLU', 'TLT', 'CAD=X'}

    # signal_structure = {}
    # for ticker in set(signal_data.values_list('ticker', flat=True)):
    #     if ticker in selected_tickers:
    #         signal_structure[ticker] = []
    #         for row in signal_data.filter(ticker__contains=ticker):
    #             signal_structure[ticker].append({
    #                 'timestamp': row.run_time,
    #                 'signal': row.signal
    #             })
    #     symbol_values[ticker].append(
    #         round(100*signal_data.filter(ticker__contains=ticker).latest('run_time').signal, 1)
    #     )

    # max_position = round(SignalTimeSeries.objects.filter(
    #     target_date__day=quarter_end_date.day,
    #     target_date__month=quarter_end_date.month,
    #     target_date__year=quarter_end_date.year,
    #     analysis_label__contains='market_continuous_multinormal_functional.r .max_position'
    # ).latest('run_time').signal*100, 1)
    
    return render(request, 'UserInterface/index.htm', {
        'current_quad_return': current_quad_return,
        'prior_quad_return': prior_quad_return,
        'daily_return': weekly_return,

        'quad_performance': quad_performance,
        'prior_quad_performance': prior_quad_performance,
        'prior_error_percentiles': prior_error_percentiles,
        'current_regressions': current_regressions,
        'prior_regressions': prior_regressions,
        'performance_change': performance_change,

        'quad_allocations': quad_allocations,
        'latest_date': latest_date,
        'target_value': net_liquidating_value,
        'data_updated': data_updated,
        'symbol_values': symbol_values,
        'lookback': lookback,

        'roc_data': quad_guesses,
        'latest_gdp_origin': latest_gdp_origin,
        'latest_gdp_sigma': latest_gdp_sigma,
        'latest_cpi_origin': latest_cpi_origin,
        'latest_cpi_sigma': latest_cpi_sigma,

        'current_quad_start': current_quad_start,
        'prior_quad_start': prior_quad_start,
        'prior_quad_end': prior_quad_end,

        'current_quad': current_quad_guess,
        'prior_quad': prior_quad_guess,

        'GOOGLE_ID': settings.GOOGLE_ID,

        # 'signal_data': signal_structure,
        #'max_position': max_position
    })
