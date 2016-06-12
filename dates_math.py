from __future__ import absolute_import
from datetime import timedelta


DAYS_IN_YEAR = 365.2425


def dates_diff_years(d1, d2):
    return (d1 - d2).days / DAYS_IN_YEAR


def add_years_to_date(d, y):
    return d + timedelta(days=y * DAYS_IN_YEAR)
