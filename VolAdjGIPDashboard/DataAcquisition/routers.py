from DataAcquisition.models import SignalTimeSeries

ROUTED_MODELS = [SignalTimeSeries]

class DBRouter(object):
    def db_for_read(self, model, **hints):
        if model in ROUTED_MODELS:
            return 'analysis'
        return None