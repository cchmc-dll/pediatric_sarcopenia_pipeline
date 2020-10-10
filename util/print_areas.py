from l3finder import ingest
import run_sma_experiment
import tables

subjects = ingest.find_subjects("/Users/jamescastiglione/git/jac241/Muscle_Segmentation/datasets/combined_dataset")
axial_series_by_subject_id = {}

for subject in subjects:
    all_series = subject.find_series()

    for series in all_series:
        try:
            if series.orientation == 'axial':
                axial_series_by_subject_id[subject.id_] = series
        except ingest.UnknownOrientation as e:
            print("Unknown orientation for ", e.series.series_path)


ds = tables.open_file("/Users/jamescastiglione/research/combined_205_fixed_checked_2020-02-18.h5")
results = []
for subject_id, truth in zip(ds.root.subject_ids, ds.root.truth):
    sma = run_sma_experiment.calculate_sma_for_series_and_mask(
        series=axial_series_by_subject_id[subject_id.decode('utf-8')],
        mask=truth[0]
    )
    results.append(sma)

for sma in results:
    print(f"{sma.subject_id}")

for sma in results:
    print(sma.area_mm2)
