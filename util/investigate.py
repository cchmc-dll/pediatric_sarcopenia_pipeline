def load_subject_ids_to_investigate(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

