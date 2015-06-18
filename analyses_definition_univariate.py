from utils import scorer_auc, scorer_angle, scorer_spearman
from toolbox.utils import evoked_subtract, evoked_spearman  # , evoked_vtest


def format_analysis(contrast):
    """This functions takes the contrasts defined for decoding  and format it
    so as to be usable by the univariate scripts

    We need to homogeneize the two types of analysis definitions
     """
    name = contrast['name']
    if contrast['scorer'] == scorer_spearman:
        operator = evoked_spearman
    elif contrast['scorer'] == scorer_auc:
        operator = evoked_subtract
    elif contrast['scorer'] == scorer_angle:
        # TODO evoked_vtest
        return
    # exclude
    exclude = dict()
    for exclude_ in contrast['exclude']:
        cond = exclude_['cond']
        exclude[cond] = exclude_['values']

    # include
    conditions = list()
    cond = contrast['include']['cond']
    for value in contrast['include']['values']:
        include_ = dict()
        include_[cond] = value
        conditions.append(dict(name=cond + str(value), include=include_,
                               exclude=exclude))
    analysis = dict(name=name, operator=operator, conditions=conditions)
    return analysis
