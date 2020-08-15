DISALLOWED_TERMS = [
    "bone",
    "lung",
    "cardiac",
    "kidney",
    "segment",
    "arterial",
    "venous",
    "delay",
    "renal",
    "kidney",
]


def _name_contains_disallowed_term(series):
    name = series.series_name.lower()
    return any(term in name for term in DISALLOWED_TERMS)


def filter_axial_series(axial_series):
    def meets_criteria(ax):
        try:
            return all([
                ax.slice_thickness in [3.0, 5.0],
                not _name_contains_disallowed_term(ax),
            ])
        except AttributeError:
            return False
    # Must be 5.0 or 3.0 slice thickness for now
    return [ax for ax in axial_series if meets_criteria(ax)]


def filter_sagittal_series(sagittal_series):
    def meets_criteria(sag):
        return not _name_contains_disallowed_term(sag)

    return [sag for sag in sagittal_series if meets_criteria(sag)]
