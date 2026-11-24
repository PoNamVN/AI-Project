"""Show three candidate Tesla models for a local photograph."""

import argparse
import json

from pathlib import Path
import numpy as np
import tensorflow as tf

from src.data_audit import MODELS
from src.train_transfer import prepare_image


def predict(run, image):
    metadata = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    labels = metadata['class_ids']
    if (metadata.get('input_size') != [224, 224] or len(set(labels)) != len(labels)
            or len(labels) < 2 or any(label not in MODELS for label in labels)):
        raise ValueError('Expected a 224px Tesla transfer model with valid class metadata')
    model = tf.keras.models.load_model(run / 'model.keras')
    if model.input_shape[1:] != (224, 224, 3) or model.output_shape[-1] != len(labels):
        raise ValueError('Model and metadata do not match')
    scores = model.predict(prepare_image(image)[None], verbose=0)[0]
    return {
        'candidates': [{'class_id': labels[i], 'model': MODELS[labels[i]], 'ranking_score': float(scores[i])}
                       for i in np.argsort(scores)[::-1][:3]],
        'review_required': True,
        'note': 'Uncalibrated scores among supported Tesla classes. Confirm visually; unsupported cars may receive a Tesla label.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--image', required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.run, args.image), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
