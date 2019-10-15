from django.shortcuts import render

from DataAcquisition.models import SecurityHistory
import datetime
import pandas as pd

def index(request, net_liquidating_value=10000):
    quad_allocation = {
        1: ['QQQ',],
        2: ['XLF', 'XLI', 'QQQ'],
        3: ['GLD',],
        4: ['XLU', 'TLT']
    }

    current_quarter_return = dict()
    prior_quarter_return = dict()
    quad_allocations = dict()

    latest_date = SecurityHistory.objects.latest('date').date

    for quad in quad_allocation:
        current_quarter_return[quad] = round(SecurityHistory.quarter_return(quad_allocation[quad], datetime.date.today())*100,1)
        prior_quarter_return[quad] = round(SecurityHistory.quarter_return(quad_allocation[quad], datetime.date.today() + pd.offsets.QuarterEnd()*0 - pd.offsets.QuarterEnd())*100, 1)
        quad_allocations[quad] = SecurityHistory.equal_volatility_position(quad_allocation[quad], target_value=net_liquidating_value)

    return render(request, 'UserInterface/index.htm', {
        'current_quarter_return': current_quarter_return,
        'prior_quarter_return': prior_quarter_return,
        'quad_allocations': quad_allocations,
        'latest_date': latest_date,
        'target_value': net_liquidating_value
    })
