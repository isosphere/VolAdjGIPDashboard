import datetime

import numpy as np
import pandas as pd
from django.shortcuts import render

from DataAcquisition.models import AlphaVantageHistory, YahooHistory


def index(request, default_net_liquidating_value=10000, lookback=252, default_currency='USD'):
    net_liquidating_value = request.POST.get('value', default_net_liquidating_value)
    currency = request.POST.get('currency', default_currency)

    if currency not in ('USD', 'CAD'):
        currency = default_currency
    
    try:
        net_liquidating_value = int(net_liquidating_value)
    except ValueError:
        net_liquidating_value = default_net_liquidating_value
    
    if currency == 'CAD':
        latest_rate = AlphaVantageHistory.objects.filter(ticker='USD.CAD').latest('date').close_price
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
            dataframe = YahooHistory.dataframe(ticker=symbol, lookback=lookback)
           
            # compute realized vol
            dataframe["log_return"] = np.log(dataframe.close_price) - np.log(dataframe.close_price.shift(1))
            dataframe["realized_vol"] = dataframe.log_return.rolling(lookback).std(ddof=0)

            latest_close, realized_vol = dataframe.iloc[-1].close_price, dataframe.iloc[-1].realized_vol          
            symbol_data.realized_volatility = realized_vol
            symbol_data.save()

        symbol_values[symbol] = (
            round(symbol_data.close_price, 2), 
            round(100*symbol_data.realized_volatility, 1), 
            round(symbol_data.close_price*symbol_data.realized_volatility, 2)
        )

    current_quarter_return = dict()
    prior_quarter_return = dict()
    quad_allocations = dict()
    
    data_updated = YahooHistory.objects.latest('updated').updated

    for quad in quad_allocation: 
        current_quarter_return[quad] = round(
            YahooHistory.quarter_return(
                tickers=quad_allocation[quad], 
                date_within_quarter=datetime.date.today()
            )*100,
            ndigits=1
        )
        prior_quarter_return[quad] = round(
            YahooHistory.quarter_return(
                tickers=quad_allocation[quad], 
                date_within_quarter=datetime.date.today() + pd.offsets.QuarterEnd()*0 - pd.offsets.QuarterEnd()
            )*100,
            ndigits=1
        )
        quad_allocations[quad] = YahooHistory.equal_volatility_position(quad_allocation[quad], target_value=net_liquidating_value)

    net_liquidating_value = round(net_liquidating_value, 0)

    return render(request, 'UserInterface/index.htm', {
        'current_quarter_return': current_quarter_return,
        'prior_quarter_return': prior_quarter_return,
        'quad_allocations': quad_allocations,
        'latest_date': latest_date,
        'target_value': net_liquidating_value,
        'data_updated': data_updated,
        'symbol_values': symbol_values,
        'lookback': lookback
    })
