import csv

import numpy as np

from datetime import datetime

import model as model_config

from data_utils import load as load_data, extract_features
from adversarial_tools import ForwardGradWrapper, adversarial_paraphrase, \
        _stats_probability_shifts


model_path = './data/model.dat'
adversarial_texts_path = './data/adversarial_texts.csv'


def clean(text):
    '''
    Clean non-unicode characters
    '''
    return ''.join([i if ord(i) < 128 else ' ' for i in str(text)])


if __name__ == '__main__':
    test_samples_cap = 2000

    # Load Twitter gender data
    (_, _, X_test, y_test), (docs_train, docs_test, _) = \
            load_data('twitter_gender_data', from_cache=False)

    # Load model from weights
    model = model_config.build_model()
    model.load_weights(model_path)

    # Initialize the class that computes forward derivatives
    grad_guide = ForwardGradWrapper(model)

    # Calculate accuracy on test examples
    preds = model.predict_classes(X_test[:test_samples_cap, ]).squeeze()
    accuracy = np.mean(preds == y_test[:test_samples_cap])
    print('Model accuracy on test:', accuracy)

    # Choose some female tweets
    female_indices, = np.where(y_test[:test_samples_cap] == 0)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    adversarial_text_data = []
    adversarial_preds = np.array(preds)

    for index, doc in enumerate(docs_test[:test_samples_cap]):
        if y_test[index] == 0 and preds[index] == 0:
            # If model prediction is correct, and the true class is female,
            # craft adversarial text
            adv_doc, (y, adv_y) = adversarial_paraphrase(
                    doc, grad_guide, target=1)

            pred = np.round(adv_y)
            if pred != preds[index]:
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            adversarial_preds[index] = pred
            adversarial_text_data.append({
                'index': index,
                'doc': clean(doc),
                'adv': clean(adv_doc),
                'success': pred != preds[index],
                'confidence': y,
                'adv_confidence': adv_y
            })

    print('Model accuracy on adversarial examples:',
            np.mean(adversarial_preds == y_test[:test_samples_cap]))
    print('Fooling success rate:',
            successful_perturbations / (successful_perturbations + failed_perturbations))
    print('Average probability shift:', np.mean(
            np.array(_stats_probability_shifts)))

    # Save resulting docs in a CSV file
    with open(adversarial_texts_path, 'w') as handle:
        writer = csv.DictWriter(handle,
                fieldnames=adversarial_text_data[0].keys())
        writer.writeheader()
        for item in adversarial_text_data:
            writer.writerow(item)
