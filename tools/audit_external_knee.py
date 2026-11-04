"""Audit public fastMRI+ annotations without inventing twelve-finding targets."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(annotations, reviewed_files):
    frame = pd.read_csv(annotations)
    required = {'file', 'slice', 'study_level', 'x', 'y', 'width', 'height', 'label'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing fastMRI+ annotation columns')
    reviewed = pd.read_csv(reviewed_files, header=None, names=['file'])['file']
    if reviewed.isna().any() or reviewed.duplicated().any():
        raise ValueError('Reviewed file IDs must be unique and nonempty')
    if frame[['file', 'label']].isna().any().any():
        raise ValueError('Missing annotation identity or label')
    missing = set(frame.file) - set(reviewed)
    if missing:
        raise ValueError(f'{len(missing)} annotated files absent from review inventory')
    counts = frame.groupby('label').agg(annotation_rows=('file', 'size'),
                                       unique_volumes=('file', 'nunique'))
    return {
        'dataset': 'fastMRI+',
        'source': 'https://github.com/microsoft/fastmri-plus',
        'input_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in (annotations, reviewed_files)},
        'reviewed_volumes': int(len(reviewed)),
        'annotated_volumes': int(frame.file.nunique()),
        'reviewed_without_annotations': int(len(set(reviewed) - set(frame.file))),
        'annotation_rows': int(len(frame)),
        'per_label': counts.to_dict(orient='index'),
        'training_images_acquired': False,
        'direct_competition_targets_generated': False,
        'policy': {
            'Meniscus Tear': 'auxiliary localization; compartment unspecified',
            'Cartilage - Partial Thickness loss/defect': 'auxiliary cartilage localization; not OA',
            'Cartilage - Full Thickness loss/defect': 'auxiliary cartilage localization; not OA',
            'Bone-Fracture/Contusion/dislocation': 'combined auxiliary class; do not split diagnoses',
            'Periarticular cysts': 'not an exact Baker cyst label',
            'unannotated': 'unknown for competition targets; never synthesize negatives',
        },
        'orientation_note': 'Source labeling conversion flips pixel arrays vertically; transform boxes with images.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--reviewed-files', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.annotations, args.reviewed_files)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in (
        'reviewed_volumes', 'annotated_volumes', 'annotation_rows')}))


if __name__ == '__main__':
    main()
