"""Selection reaches safety stock and cartons on a controlled single-SKU relay."""

from clinicstock.carton_budget import evaluate as allocate_cartons
from clinicstock.demand_smoothing.model_selection import selection_days, select_for_record
from clinicstock.pipeline import compose_requests
from clinicstock.reorder_bands import evaluate as build_bands


STORE = '测试库点'
SKU = 'W-1'
TODAY = 56


def error(day, actual, predicted, naive):
    return {'day': day, 'actual': actual, 'predicted': predicted, 'naive': naive, 'residual': actual - predicted}


def selected_record(*, baseline_wins, critical=False):
    """One modelled record with a clear winner and a full seven-day horizon."""
    errors = []
    for day in range(TODAY):
        if baseline_wins:
            errors.append(error(day, 10, 12, 10))  # model off, baseline exact
        else:
            errors.append(error(day, 10, 10, 12))  # model exact, baseline off
    model_daily, baseline_daily = [12] * 7, [4] * 7
    record = {
        'store': STORE, 'sku': SKU, 'history_days': TODAY, 'horizon': 7,
        'training_errors': errors,
        'daily_forecast': model_daily, 'forecast_total': float(sum(model_daily)),
        'baseline_daily_forecast': baseline_daily, 'baseline_forecast_total': float(sum(baseline_daily)),
        'baseline_ready': True,
        'demand_evidence': {'cycle': [{'forecast_units': 0, 'basis': 'WEEKDAY_MEDIAN'} for _ in range(7)],
                            'history': {}, 'backtest': {}},
    }
    settled = {(STORE, SKU, row['day']) for row in errors}
    return select_for_record(record, settled_days=settled, critical=critical), settled


def plan_for(record, settled):
    residuals = selection_days(record, settled_days=settled, method=record['forecast_method'])
    validation = [dict(row, store=STORE, sku=SKU, available=True) for row in residuals]
    bands = build_bands(validation, as_of=TODAY)
    shelf = [{'sku': SKU, 'store': STORE, 'pack': 10, 'case_cost': 500, 'on_hand': 0, 'inbound': 0,
              'critical': False, 'cold_capacity': 500, 'min_term_days': 30}]
    requests = list(compose_requests(shelf, [record], bands['records']))
    orders = allocate_cartons(requests, money=100000)['records']
    return bands['records'][0], orders[0]


def test_switched_line_orders_from_the_baseline_horizon_and_baseline_residuals():
    record, settled = selected_record(baseline_wins=True)
    assert record['forecast_method'] == 'BASELINE'
    band, order = plan_for(record, settled)
    assert record['forecast_total'] == 28.0
    assert order['forecast'] == 28.0
    assert band['past_windows'] == [0] * band['effective_windows']
    assert order['validated'] and order['allocated_cases'] > 0


def test_kept_model_line_orders_from_the_model_horizon_and_model_residuals():
    record, settled = selected_record(baseline_wins=False)
    assert record['forecast_method'] == 'MODEL'
    band, order = plan_for(record, settled)
    assert record['forecast_total'] == 84.0
    assert order['forecast'] == 84.0
    assert band['past_windows'] == [0] * band['effective_windows']
    assert order['allocated_cases'] == 9


def test_critical_guard_keeps_model_horizon_despite_lower_baseline_mae():
    record, settled = selected_record(baseline_wins=True)
    # Break one week so the baseline under-estimates more than the model there,
    # but keep its overall MAE lower than the model's.
    errors = [dict(row) for row in record['training_errors']]
    errors[3] = error(3, 40, 39, 10)
    settled = {(STORE, SKU, row['day']) for row in errors}
    raw = {'store': record['store'], 'sku': record['sku'], 'history_days': record['history_days'],
           'horizon': record['horizon'], 'training_errors': errors,
           'daily_forecast': record['model_daily_forecast'], 'forecast_total': record['model_forecast_total'],
           'baseline_daily_forecast': record['baseline_daily_forecast'],
           'baseline_forecast_total': record['baseline_forecast_total'],
           'baseline_ready': record['baseline_ready'], 'demand_evidence': record['demand_evidence']}
    guarded = select_for_record(raw, settled_days=settled, critical=True)
    plain = select_for_record(raw, settled_days=settled, critical=False)

    assert plain['forecast_method'] == 'BASELINE'
    assert guarded['forecast_method'] == 'MODEL'
    assert guarded['forecast_choice_reason'] == 'CRITICAL_UNDERESTIMATE_GUARD'
    _, order = plan_for(guarded, settled)
    assert order['forecast'] == guarded['model_forecast_total'] == 84.0
