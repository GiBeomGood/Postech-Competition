def calc_profit(actual_gens: list[float], forecast_gens: list[float]):
    CAPACITY = 99.0
    facility_utilization_rate = [actual / CAPACITY for actual in actual_gens]

    filter_facility_utilization_rate = [
        utilization >= 0.1 for utilization in facility_utilization_rate
    ]

    errors = [
        abs(forecast - actual) / CAPACITY * 100
        for forecast, actual in zip(forecast_gens, actual_gens)
    ]

    target_errors = [
        error
        for error, is_filtered in zip(errors, filter_facility_utilization_rate)
        if is_filtered
    ]
    target_actual_gens = [
        actual
        for actual, is_filtered in zip(
            actual_gens, filter_facility_utilization_rate
        )
        if is_filtered
    ]

    profits = [0] * len(target_actual_gens)

    for i, error in enumerate(target_errors):
        if error <= 6:
            profits[i] = target_actual_gens[i] * 4
        elif 6 < error <= 8:
            profits[i] = target_actual_gens[i] * 3

    return profits
