import datetime

import numpy as np
import pandas as pd
from django.shortcuts import render

from DataAcquisition.models import AlphaVantageHistory, YahooHistory, QuadForecasts
from django.contrib import messages


def index(request, default_net_liquidating_value=10000, lookback=28, default_currency='USD'):
    # For our little 4-quad chart
    current_date = datetime.date.today()
    quarter_int = (current_date.month - 1) // 3 + 1 
    quarter_date = datetime.date(current_date.year, 1, 1) + pd.offsets.QuarterEnd()*quarter_int

    quad_guesses = QuadForecasts.objects.filter(quarter_end_date=quarter_date).order_by('-date')[:2].values_list('date', 'gdp_roc', 'cpi_roc')

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
        4: ['XLU', 'TLT']
    }

    symbol_values = dict()

    latest_date = YahooHistory.objects.latest('date').date

    all_symbols = list()
    for quad in quad_allocation:
        for symbol in quad_allocation[quad]:
            if symbol not in all_symbols:
                all_symbols.append(symbol)

    all_symbols.sort()

    for symbol in all_symbols:
        symbol_data = YahooHistory.objects.get(ticker=symbol, date=latest_date)

        if symbol_data.realized_volatility is None:
            dataframe = YahooHistory.dataframe(tickers=[symbol], lookback=lookback)
            dataframe["log_return"] = dataframe.groupby(level='ticker').close_price.apply(np.log) - dataframe.groupby(level='ticker').close_price.shift(1).apply(np.log)
            dataframe["realized_vol"] = dataframe.groupby(level='ticker').log_return.rolling(lookback).std(ddof=0).droplevel(0)
            dataframe = dataframe.droplevel("ticker")

            latest_close, realized_vol = dataframe.iloc[-1].close_price, dataframe.iloc[-1].realized_vol          
            symbol_data.realized_volatility = realized_vol
            symbol_data.save()

        symbol_values[symbol] = (
            round(symbol_data.close_price, 2), 
            round(100*symbol_data.realized_volatility, 2), 
            round(symbol_data.close_price*symbol_data.realized_volatility, 2)
        )
    symbol_values["USDCAD"] = (
        latest_rate,
        "--.--",
        "--.--"
    )

    # for positioning, at the bottom of our page
    quad_allocations = dict()

    # Quad Return Calculation
    current_quad = QuadForecasts.objects.latest('quarter_end_date', 'date').quad
    prior_quad_end_date = QuadForecasts.objects.exclude(quad=current_quad).latest('quarter_end_date', 'date').date

    current_quad_return = dict()
    prior_quad_return = dict()
    
    data_updated = YahooHistory.objects.latest('updated').updated

    for quad in quad_allocation:
        try_date = datetime.date.today()

        while True:
            try:
                current_quad_return[quad] = round(
                    YahooHistory.quad_return(
                        tickers=quad_allocation[quad], 
                        date_within_quad=try_date
                    )*100,
                    ndigits=1
                )
                prior_quad_return[quad] = round(
                    YahooHistory.quad_return(
                        tickers=quad_allocation[quad], 
                        date_within_quad=prior_quad_end_date
                    )*100,
                    ndigits=1
                )
                break
            except YahooHistory.DoesNotExist:
                try_date -= datetime.timedelta(days=1)

        quad_allocations[quad] = YahooHistory.equal_volatility_position(quad_allocation[quad], target_value=net_liquidating_value)

    net_liquidating_value = round(net_liquidating_value, 0)

    return render(request, 'UserInterface/index.htm', {
        'current_quad_return': current_quad_return,
        'prior_quad_return': prior_quad_return,
        'quad_allocations': quad_allocations,
        'latest_date': latest_date,
        'target_value': net_liquidating_value,
        'data_updated': data_updated,
        'symbol_values': symbol_values,
        'lookback': lookback,
        'roc_data': quad_guesses,
        'prior_quad_end': prior_quad_end_date
    })
