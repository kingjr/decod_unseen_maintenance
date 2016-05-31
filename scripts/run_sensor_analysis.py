"""Performs sensor analysis within each subjects separately"""
from base import nested_analysis, load, save
from config import subjects
from conditions import analyses

for subject in subjects:
    print('load %s' % subject)

    epochs = load('epochs', subject=subject)
    events = load('behavior', subject=subject)

    # Apply each analysis
    for analysis in analyses:
        print(analysis['name'])

        # This functions computes nested contrast and return the effect size
        # for each level of comparison.
        # FIXME : This is an overkill here since we only apply 1 level analysis
        coef, sub = nested_analysis(
            epochs._data, events, analysis['condition'],
            function=analysis.get('erf_function', None),
            query=analysis.get('query', None),
            single_trial=analysis.get('single_trial', False),
            y=analysis.get('y', None))

        evoked = epochs.average()
        evoked.data = coef

        # Save all_evokeds
        save([evoked, sub, analysis], 'evoked', subject=subject,
             analysis=analysis['name'])
