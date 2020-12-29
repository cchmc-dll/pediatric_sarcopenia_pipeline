import pickle

import attr


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
]


@attr.s
class SeriesExclusion:
    reason = attr.ib()
    series = attr.ib()


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

    def analyze_criteria(ax):
        exclusions = []
        try:
            if not ax.slice_thickness in [3.0, 5.0]:
                exclusions.append(
                    SeriesExclusion(
                        series=ax,
                        reason="Axial slice thickness not 3.0 or 5.0",
                    )
                )
            if _name_contains_disallowed_term(ax):
                exclusions.append(
                    SeriesExclusion(
                        series=ax,
                        reason="Axial series name had disallowed term"
                    )
                )
        except AttributeError:
            exclusions.append(
                SeriesExclusion(
                    series=ax,
                    reason="AttributeError when determining exclusions for axial series"
                )
            )
        return exclusions

    # Must be 5.0 or 3.0 slice thickness for now
    maybe_excluded_list = [analyze_criteria(ax) for ax in axial_series]

    remaining = [
        ax
        for ax, exclusions
        in zip(axial_series, maybe_excluded_list)
        if len(exclusions) == 0
    ]
    exclusions = [e for e in maybe_excluded_list if len(e) > 0]

    return remaining, exclusions


def filter_sagittal_series(sagittal_series):
    def meets_criteria(sag):
        return not _name_contains_disallowed_term(sag)

    def analyze_criteria(sag):
        exclusions = []
        if _name_contains_disallowed_term(sag):
            exclusions.append(
                SeriesExclusion(
                    series=sag,
                    reason="Axial series name had disallowed term"
                )
            )
        return exclusions

    maybe_excluded_list = [analyze_criteria(sag) for sag in sagittal_series]

    remaining = [
        sag
        for sag, exclusions
        in zip(sagittal_series, maybe_excluded_list)
        if len(exclusions) == 0
    ]
    exclusions = [e for e in maybe_excluded_list if len(e) > 0]

    return remaining, exclusions




def load_series_to_skip_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def remove_series_to_skip(series_to_skip, input_series):
    series_paths_to_skip = set(s.series_path for s, _ in series_to_skip)

    return [s for s in input_series if s.series_path not in series_paths_to_skip]


