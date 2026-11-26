"""Choose the seven-day forecast behind each replenishment line.

The weekday/Croston model stays the default. The prior-week baseline may
replace it only when it is compared on complete, settled, non-overlapping
weeks in which both methods produced a value, and only when it has the lower
raw mean absolute error. Critical items keep the model as soon as one
comparable week shows the baseline underestimating use by more units.
"""

from math import fsum

MODEL = 'MODEL'
BASELINE = 'BASELINE'
MINIMUM_COMPARISON_WEEKS = 4


def paired_complete_weeks(errors, horizon):
    """Return complete weeks where both methods have a value on every day.

    Weeks are counted on consecutive calendar days and never overlap. A day
    without a paired value, or a gap in settlement, restarts the window so a
    partial tail can never be compared as a full week.
    """
    weeks = []
    current = []
    for row in errors:
        if row['predicted'] is None or row['naive'] is None:
            current = []
            continue
        if current and row['day'] != current[-1]['day'] + 1:
            current = []
        current.append(row)
        if len(current) == horizon:
            weeks.append(tuple(current))
            current = []
    return weeks


def week_underestimate(week, field):
    return fsum(max(0.0, row['actual'] - row[field]) for row in week)


def compare_weeks(weeks):
    days = [row for week in weeks for row in week]
    if not days:
        return None
    return {
        'model_mae': fsum(abs(row['actual'] - row['predicted']) for row in days) / len(days),
        'baseline_mae': fsum(abs(row['actual'] - row['naive']) for row in days) / len(days),
    }


def choose(errors, *, horizon, model_ready, baseline_ready, critical):
    """Select a method using unrounded errors and return its evidence."""
    weeks = paired_complete_weeks(errors, horizon)
    scores = compare_weeks(weeks)
    guard_weeks = [
        {'start': week[0]['day'], 'end': week[-1]['day']}
        for week in weeks
        if week_underestimate(week, 'naive') > week_underestimate(week, 'predicted')
    ]
    evidence = {
        'critical': bool(critical),
        'comparable_weeks': len(weeks),
        'minimum_weeks': MINIMUM_COMPARISON_WEEKS,
        'model_mae': scores['model_mae'] if scores else None,
        'baseline_mae': scores['baseline_mae'] if scores else None,
        'baseline_horizon_ready': bool(baseline_ready),
        'critical_guard_weeks': guard_weeks,
    }
    if not model_ready:
        return MODEL, 'SOURCE_HISTORY_TOO_SPARSE', evidence
    if not baseline_ready:
        return MODEL, 'BASELINE_HORIZON_UNAVAILABLE', evidence
    if len(weeks) < MINIMUM_COMPARISON_WEEKS:
        return MODEL, 'INSUFFICIENT_COMPARABLE_WEEKS', evidence
    if scores['baseline_mae'] < scores['model_mae']:
        if critical and guard_weeks:
            return MODEL, 'CRITICAL_UNDERESTIMATE_GUARD', evidence
        return BASELINE, 'BASELINE_LOWER_ERROR', evidence
    return MODEL, 'MODEL_NOT_WORSE', evidence


def settled_errors(record, settled_days):
    """Keep only backtest days whose outcome was a settled, complete day."""
    identity = record['store'], record['sku']
    return [row for row in record['training_errors'] if identity + (row['day'],) in settled_days]


def select_for_record(record, *, settled_days, critical):
    """Attach the chosen forecast and its evidence to one modelling record.

    ``daily_forecast``/``forecast_total`` become the selected horizon, so
    safety stock and cartons consume exactly the method that was chosen. The
    model's own horizon, the baseline horizon and the day-level backtest stay
    on the record for audit, and the evidence block is rebuilt for the selected
    horizon so its per-day basis reflects what is actually ordered.
    """
    errors = settled_errors(record, settled_days)
    method, reason, evidence = choose(
        errors,
        horizon=record['horizon'],
        model_ready=record['forecast_total'] is not None,
        baseline_ready=record.get('baseline_ready', False),
        critical=critical,
    )
    model_daily = record['daily_forecast']
    model_total = record['forecast_total']
    baseline_daily = record['baseline_daily_forecast']
    baseline_total = record['baseline_forecast_total']
    daily, total = (baseline_daily, baseline_total) if method == BASELINE else (model_daily, model_total)
    evidence_block = {
        **record['demand_evidence'],
        'cycle': [dict(item) for item in record['demand_evidence']['cycle']],
        'forecast_method': method,
    }
    for item, value in zip(evidence_block['cycle'], daily):
        item['forecast_units'] = value
        item['forecast_method'] = method
    return record | {
        'forecast_method': method,
        'forecast_choice_reason': reason,
        'selection_evidence': evidence,
        'model_daily_forecast': model_daily,
        'model_forecast_total': model_total,
        'baseline_daily_forecast': baseline_daily,
        'baseline_forecast_total': baseline_total,
        'daily_forecast': list(daily),
        'forecast_total': total,
        'demand_evidence': evidence_block,
    }


def selection_days(record, *, settled_days, method):
    """Return settled days paired with the chosen method's one-day estimate.

    These are the residuals safety stock is rebuilt from after a switch.
    """
    identity = record['store'], record['sku']
    rows = []
    for row in record['training_errors']:
        if identity + (row['day'],) not in settled_days:
            continue
        estimate = row['naive'] if method == BASELINE else row['predicted']
        if estimate is None:
            continue
        rows.append(row | {'predicted': estimate})
    return rows

