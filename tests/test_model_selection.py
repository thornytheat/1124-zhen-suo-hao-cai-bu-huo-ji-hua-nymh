from clinicstock.demand_smoothing.model_selection import (
    BASELINE,
    MINIMUM_COMPARISON_WEEKS,
    MODEL,
    choose,
    paired_complete_weeks,
    select_for_record,
    selection_days,
    settled_errors,
)


def error(day, actual, predicted, naive):
    return {'day': day, 'actual': actual, 'predicted': predicted, 'naive': naive, 'residual': actual - predicted}


def weeks_of(rows):
    """Pad ``rows`` into weeks, one error row per consecutive day."""
    return [error(day, *values) for day, values in enumerate(rows)]


def triples(count):
    """``count`` days of (actual, model, baseline) all scoring equally."""
    return [error(day, 1, 1, 1) for day in range(count)]


def good_baseline_days(weeks=MINIMUM_COMPARISON_WEEKS, days=7, actual=10):
    """A run where prior-week repeat is exact and the model is off by one."""
    return [error(day, actual, actual + 1, actual) for day in range(weeks * days)]


def record(rows, *, daily=None, baseline_daily=None, critical=False, horizon=7):
    model_daily = daily or [5] * 7
    base_daily = baseline_daily or [4] * 7
    return {
        'store': '门诊', 'sku': 'W1',
        'training_errors': rows, 'horizon': horizon,
        'daily_forecast': list(model_daily), 'forecast_total': sum(model_daily),
        'baseline_daily_forecast': list(base_daily),
        'baseline_forecast_total': sum(base_daily) if all(v is not None for v in base_daily) else None,
        'baseline_ready': all(v is not None for v in base_daily),
        'demand_evidence': {'cycle': [{'forecast_units': 0, 'basis': 'WEEKDAY_MEDIAN'} for _ in range(7)],
                            'history': {}, 'backtest': {}},
        'as_of': 100,
        'critical': critical,
    }


def settled_every_day(rows):
    return {('门诊', 'W1', row['day']) for row in rows}


def test_complete_weeks_are_non_overlapping_and_drop_partial_tail():
    rows = triples(9)
    weeks = paired_complete_weeks(rows, 7)
    assert [week[0]['day'] for week in weeks] == [0]
    assert paired_complete_weeks(rows + [error(9, 1, 1, 1)], 7)[-1][0]['day'] == 0


def test_gap_in_settlement_prevents_a_week_being_counted():
    rows = [row for row in triples(14) if row['day'] != 6]
    assert len(paired_complete_weeks(rows, 7)) == 1


def test_baseline_replaces_model_only_on_comparable_complete_weeks():
    rows = good_baseline_days()
    method, reason, evidence = choose(rows, horizon=7, model_ready=True, baseline_ready=True, critical=False)
    assert method == BASELINE and reason == 'BASELINE_LOWER_ERROR'
    assert evidence['comparable_weeks'] == MINIMUM_COMPARISON_WEEKS
    assert evidence['baseline_mae'] < evidence['model_mae']


def test_too_few_comparable_weeks_keeps_the_current_forecast():
    rows = good_baseline_days(weeks=MINIMUM_COMPARISON_WEEKS - 1)
    method, reason, _ = choose(rows, horizon=7, model_ready=True, baseline_ready=True, critical=False)
    assert method == MODEL and reason == 'INSUFFICIENT_COMPARABLE_WEEKS'


def test_unreachable_baseline_seven_day_horizon_keeps_the_current_forecast():
    rows = good_baseline_days()
    method, reason, evidence = choose(rows, horizon=7, model_ready=True, baseline_ready=False, critical=False)
    assert method == MODEL and reason == 'BASELINE_HORIZON_UNAVAILABLE'
    assert evidence['baseline_horizon_ready'] is False


def test_critical_item_keeps_model_when_baseline_underestimates_in_one_week():
    rows = good_baseline_days()
    rows[10] = error(10, actual=20, predicted=18, naive=12)
    method, reason, evidence = choose(rows, horizon=7, model_ready=True, baseline_ready=True, critical=True)
    assert method == MODEL and reason == 'CRITICAL_UNDERESTIMATE_GUARD'
    assert evidence['critical_guard_weeks'] == [{'start': 7, 'end': 13}]


def test_noncritical_item_with_same_evidence_may_still_switch():
    rows = good_baseline_days()
    rows[10] = error(10, actual=20, predicted=18, naive=12)
    method, reason, _ = choose(rows, horizon=7, model_ready=True, baseline_ready=True, critical=False)
    assert method == BASELINE and reason == 'BASELINE_LOWER_ERROR'


def test_tie_or_model_advantage_keeps_current_forecast():
    tied = [error(day, 10, 10, 10) for day in range(28)]
    method, reason, _ = choose(tied, horizon=7, model_ready=True, baseline_ready=True, critical=False)
    assert method == MODEL and reason == 'MODEL_NOT_WORSE'


def test_raw_precision_decides_even_when_rounded_scores_agree():
    rows = [error(day, 10, 10.001, 10.0) for day in range(28)]
    method, reason, evidence = choose(rows, horizon=7, model_ready=True, baseline_ready=True, critical=False)
    assert round(evidence['model_mae'], 2) == round(evidence['baseline_mae'], 2)
    assert method == BASELINE and reason == 'BASELINE_LOWER_ERROR'


def test_switch_uses_the_baseline_horizon_and_baseline_residuals():
    rows = good_baseline_days()
    selected = select_for_record(record(rows), settled_days=settled_every_day(rows), critical=False)
    assert selected['forecast_method'] == BASELINE
    assert selected['forecast_total'] == selected['baseline_forecast_total']
    assert selected['daily_forecast'] == selected['baseline_daily_forecast']
    assert selected['demand_evidence']['forecast_method'] == BASELINE
    assert [item['forecast_units'] for item in selected['demand_evidence']['cycle']] == selected['baseline_daily_forecast']


def test_model_choice_keeps_the_model_horizon_and_original_evidence():
    rows = [error(day, 10, 10, 10) for day in range(28)]
    selected = select_for_record(record(rows, daily=[7] * 7, baseline_daily=[3] * 7),
                                 settled_days=settled_every_day(rows), critical=False)
    assert selected['forecast_method'] == MODEL
    assert selected['forecast_total'] == 49
    assert selected['daily_forecast'] == [7] * 7
    assert selected['demand_evidence']['cycle'][0]['forecast_units'] == 7


def test_current_forecast_kept_when_model_has_too_little_history():
    method, reason, _ = choose([], horizon=7, model_ready=False, baseline_ready=True, critical=False)
    assert method == MODEL and reason == 'SOURCE_HISTORY_TOO_SPARSE'


def test_days_without_a_paired_baseline_value_are_not_scored():
    rows = good_baseline_days()
    rows[3] = error(3, 10, 11, None)
    weeks = paired_complete_weeks(rows, 7)
    assert all(row['day'] != 3 for week in weeks for row in week)
    assert len(weeks) == MINIMUM_COMPARISON_WEEKS - 1


def test_settled_days_filter_unsettled_outcomes_before_scoring():
    rows = good_baseline_days()
    rows[2] = error(2, 30, 99, 0)  # would flip the decision if it counted
    settled = {('门诊', 'W1', row['day']) for row in rows if row['day'] != 2}
    kept = settled_errors(record(rows), settled)
    assert all(row['day'] != 2 for row in kept)
    assert len(kept) == len(rows) - 1


def test_selected_residuals_follow_the_chosen_method_only():
    rows = good_baseline_days()
    rec = select_for_record(record(rows), settled_days=settled_every_day(rows), critical=False)
    residuals = selection_days(rec, settled_days=settled_every_day(rows), method=rec['forecast_method'])
    assert len(residuals) == len(rows)
    assert all(row['predicted'] == row['naive'] for row in residuals)


def test_reorder_band_input_after_switch_uses_baseline_residuals():
    from clinicstock.reorder_bands.service import evaluate

    rows = good_baseline_days()
    rec = select_for_record(record(rows), settled_days=settled_every_day(rows), critical=False)
    residuals = selection_days(rec, settled_days=settled_every_day(rows), method=rec['forecast_method'])
    validation = [dict(row, store=rec['store'], sku=rec['sku'], available=True) for row in residuals]
    band, = evaluate(validation, as_of=rec['as_of'], horizon=7, minimum_windows=MINIMUM_COMPARISON_WEEKS)['records']
    assert band['past_windows'] == [0] * MINIMUM_COMPARISON_WEEKS
    assert band['cycle_buffer'] == 0


def test_selection_uses_the_modelling_horizon_not_a_fixed_week():
    rows = [error(day, 10, 11, 10) for day in range(5 * 3)]
    rec = select_for_record(record(rows, daily=[5] * 3, baseline_daily=[4] * 3, horizon=3),
                            settled_days=settled_every_day(rows), critical=False)
    assert rec['selection_evidence']['comparable_weeks'] == 5
    assert rec['forecast_method'] == BASELINE
    assert rec['forecast_total'] == 12
