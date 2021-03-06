import datetime

import numpy as np
import pandas as pd

from DataAcquisition.models import AlphaVantageHistory, YahooHistory, QuadForecasts, QuadReturn, CommitmentOfTraders
from django.db.models import F
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render
from django.http import Http404

def quad_performance(request, label):
    label = label.replace(' ', '')
    latest_date = YahooHistory.objects.latest('date').date
    
    current_quad_forecast = QuadForecasts.objects.filter(quarter_end_date=latest_date + pd.tseries.offsets.QuarterEnd(n=0)).latest('date')
    current_quad, quarter_end_date = current_quad_forecast.quad, current_quad_forecast.quarter_end_date
    prior_quad_end_date = (quarter_end_date - pd.tseries.offsets.QuarterEnd(n=1)).date()
    current_quad_start = prior_quad_end_date + datetime.timedelta(days=1)
    prior_quad_start = (prior_quad_end_date - pd.tseries.offsets.QuarterEnd(n=1) + datetime.timedelta(days=1)).date()

    # time series data for quad return charts
    quad_returns = QuadReturn.objects.filter(quarter_end_date=quarter_end_date, label=label).order_by('label', 'data_end_date').annotate(score=F('quad_return')/F('quad_stdev'))
    prior_quad_returns = QuadReturn.objects.filter(quarter_end_date=prior_quad_end_date, label=label).order_by('label', 'data_end_date').annotate(score=F('quad_return')/F('quad_stdev'))

    if not quad_returns:
        raise Http404(f"Label {label} does not exist.")

    quad_performance = list()
    for ticker_lookup, date, score in quad_returns.values_list('label', 'data_end_date', 'score'):
        quad_performance.append(((date-current_quad_start).days, round(score, 2)))

    prior_quad_performance = list()
    for ticker_lookup, date, score in prior_quad_returns.values_list('label', 'data_end_date', 'score'):
        prior_quad_performance.append(((date-prior_quad_start).days, round(score, 2)))

    latest_performance = quad_returns.latest('data_end_date').data_end_date

    current_performance = quad_returns.get(data_end_date=latest_performance).score
    prior_performance = quad_returns.exclude(data_end_date=latest_performance).latest('data_end_date').score
    performance_change = round(current_performance / prior_performance, ndigits=1)
        
    # fix for negative comparisons showing positive change
    if current_performance < prior_performance and performance_change > 0:
        performance_change *= -1
    
    return render(request, 'UserInterface/performance.htm', {
        'label': label,
        'performance_change': performance_change,
        'quad_performance': quad_performance,
        'prior_quad_performance': prior_quad_performance
    })

def all_symbol_summary(quad_allocation, latest_date):
    symbol_values = dict()

    for group in (YahooHistory, AlphaVantageHistory):
        if group.__name__ == 'YahooHistory':
            all_symbols = list()
            for quad in quad_allocation:
                for symbol in quad_allocation[quad]:
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)

            all_symbols += ['XLV', 'SHY', 'EDV', 'IWM', 'PSP', 'RSP', 'JNK', 'FXB', 'EWG', 'EWA', 'ITB', 'TIP', 'VTI', 'BND', 'XLE']
        else:
            all_symbols = list(AlphaVantageHistory.objects.all().values_list('ticker', flat=True).distinct())
        
        all_symbols.sort()

        for symbol in all_symbols:
            try:
                symbol_data = group.objects.get(ticker=symbol, date=latest_date)
            except group.DoesNotExist:
                group.update(tickers=all_symbols)
                try:
                    symbol_data = group.objects.get(ticker=symbol, date=latest_date)
                except group.DoesNotExist:
                    symbol_values[symbol] = (
                        'N/A',
                        '--.--',
                        '--.--',
                        '--.--',
                        '--.--',
                        group.__name__ + '_' + symbol
                    )
                    continue

            prior_week_ref = group.objects.filter(ticker=symbol).latest('date').date - datetime.timedelta(weeks=1)
            prior_week = prior_week_ref.isocalendar()[1]
            last_week = group.objects.filter(ticker=symbol, date__week=prior_week).latest('date')
            last_week_date, last_week_val = last_week.date, last_week.close_price

            # get realized vol as of last week - don't let the buy/sell targets move with current vol
            last_week_vol = group.objects.get(ticker=symbol, date=last_week_date).realized_volatility
            
            try:
                current_performance = QuadReturn.objects.filter(label=f"{group.__name__}_{symbol}").latest('quarter_end_date', 'data_end_date')
                if current_performance:
                    current_performance = current_performance.quad_return / current_performance.quad_stdev
            except QuadReturn.DoesNotExist:
                current_performance = None

            if last_week_vol is not None:
                symbol_values[symbol] = (
                    round(symbol_data.close_price, 2), 
                    round(100*last_week_vol, 2), 
                    round(last_week_val * ( 1 - last_week_vol), 2),
                    round(last_week_val * ( 1 + last_week_vol), 2),
                    int(round(100*(symbol_data.close_price - last_week_val*(1 - last_week_vol)) / ( last_week_val * ( 1 + last_week_vol) - last_week_val * ( 1 - last_week_vol)), 0)),
                    '--.--' if not current_performance else round(current_performance, 2),
                    group.__name__ + '_' + symbol
                )
            else:
                symbol_values[symbol] = (
                    round(symbol_data.close_price, 2), 
                    '--.--', 
                    '--.--',
                    '--.--',
                    '--.--',
                    '--.--' if not current_performance else round(current_performance, 2),
                    group.__name__ + '_' + symbol
                )

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
    # Commitment of Traders
    latest_cot_date = CommitmentOfTraders.objects.latest('date').date
    cot_data = CommitmentOfTraders.objects.filter(date=latest_cot_date).order_by('symbol')

    # For our little 4-quad chart
    current_date = datetime.date.today()
    quarter_int = (current_date.month - 1) // 3 + 1 
    quarter_date = datetime.date(current_date.year, 1, 1) + pd.offsets.QuarterEnd(n=1)*quarter_int

    quad_guesses = QuadForecasts.objects.filter(quarter_end_date=quarter_date).order_by('-date')[:3].values_list('date', 'gdp_roc', 'cpi_roc')

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
    latest_rate = AlphaVantageHistory.objects.filter(ticker='USD.CAD').latest('date').close_price
    if currency == 'CAD':
        net_liquidating_value /= latest_rate

    quad_allocation = {
        1: ['QQQ',],
        2: ['XLF', 'XLI', 'QQQ'],
        3: ['GLD',],
        4: ['XLU', 'TLT', 'UUP']
    }

    daily_return = dict()
    for quad in quad_allocation:
        current_daily_return = YahooHistory.daily_return(quad_allocation[quad])
        daily_return[quad] = current_daily_return*100 if current_daily_return is not None else None

    latest_date = YahooHistory.objects.latest('date').date

    symbol_values = all_symbol_summary(quad_allocation, latest_date)

    # for positioning, at the bottom of our page
    quad_allocations = dict()
    for quad in quad_allocation:
        quad_allocations[quad] = YahooHistory.equal_volatility_position(quad_allocation[quad], target_value=net_liquidating_value)

    # Quad Return Calculation
    current_quad_forecast = QuadForecasts.objects.filter(quarter_end_date=latest_date + pd.tseries.offsets.QuarterEnd(n=0)).latest('date')
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

    net_liquidating_value = round(net_liquidating_value, 0)

    # time series data for quad return charts
    quad_labels = ('YahooHistory_QQQ', 'YahooHistory_QQQ,XLF,XLI', 'YahooHistory_GLD', 'YahooHistory_TLT,UUP,XLU')
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
        quad = quad_ticker_lookup[ticker_lookup]

        if quad not in quad_performance:
            quad_performance[quad] = list()
        
        quad_performance[quad].append(((date-current_quad_start).days, round(score, 2)))

    prior_quad_performance = dict()
    for ticker_lookup, date, score in prior_quad_returns.values_list('label', 'data_end_date', 'score'):
        quad = quad_ticker_lookup[ticker_lookup]

        if quad not in prior_quad_performance:
            prior_quad_performance[quad] = list()
        
        prior_quad_performance[quad].append(((date-prior_quad_start).days, round(score, 2)))

    performance_change = dict()
    latest_performance = quad_returns.latest('data_end_date').data_end_date
    for lookup in quad_ticker_lookup:
        quad = quad_ticker_lookup[lookup]
        current_performance = quad_returns.get(label=lookup, data_end_date=latest_performance).score
        prior_performance = quad_returns.exclude(data_end_date=latest_performance).filter(label=lookup).latest('data_end_date').score
        performance_change[quad] = round(current_performance - prior_performance, ndigits=1)

    return render(request, 'UserInterface/index.htm', {
        'current_quad_return': current_quad_return,
        'prior_quad_return': prior_quad_return,
        'daily_return': daily_return,

        'quad_performance': quad_performance,
        'prior_quad_performance': prior_quad_performance,
        'performance_change': performance_change,

        'quad_allocations': quad_allocations,
        'latest_date': latest_date,
        'target_value': net_liquidating_value,
        'data_updated': data_updated,
        'symbol_values': symbol_values,
        'lookback': lookback,
        'roc_data': quad_guesses,

        'current_quad_start': current_quad_start,
        'prior_quad_start': prior_quad_start,
        'prior_quad_end': prior_quad_end,

        'latest_cot_date': latest_cot_date,
        'cot_data': cot_data,
        'GOOGLE_ID': settings.GOOGLE_ID
    })
