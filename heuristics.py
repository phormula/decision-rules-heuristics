import math


def N(T, a=None):
    if a is None:
        return T.shape[0]
    else:
        decision_class = list(T.columns)[-1]
        return T[T[decision_class] == a].shape[0]


def M(T, a=None):
    return N(T) - N(T, a)


def alpha(T, subtable, a):
    return N(T, a)-N(subtable, a)


def beta(T, subtable, a):
    return M(T, a) - M(subtable, a)


def Poly(subtable, decision_value, T):
    return beta(T, subtable, decision_value)/(alpha(T, subtable, decision_value)+1)


def Log(subtable,  decision_value, T):
    return beta(T, subtable, decision_value)/math.log2(alpha(T, subtable, decision_value)+2)


def RM(subtable, decision_value, T=None):
    NTj1 = N(subtable)
    NTj1a = N(subtable, decision_value)

    return (NTj1 - NTj1a)/NTj1


def heuristic(df, row_index, heuristic_type):
    metric_func = min if heuristic_type.__name__ == 'RM' else max
    headings = list(df.columns)
    features = headings[:-1]
    decision_class = headings[-1]
    decision_value = df.loc[row_index, decision_class]
    rule = []

    while df[decision_class].nunique() > 1:
        subtables = []
        algorithm_vals = []
        for feature in features:
            subtable = df[df[feature] == df.loc[row_index, feature]]
            algorithm_val = heuristic_type(subtable, decision_value, df)
            algorithm_vals.append(algorithm_val)
            subtables.append((feature, subtable))
            # display(feature,algorithm_val,subtable)

        selected_index = algorithm_vals.index(metric_func(algorithm_vals))
        selected_feature, selected_subtable = subtables[selected_index]
        feature_value = selected_subtable.loc[row_index, selected_feature]
        rule.append({selected_feature: str(feature_value)})
        df = selected_subtable.drop(selected_feature, axis=1)
        headings = list(df.columns)
        features = headings[:-1]

    rule.append({'d': str(decision_value)})

    return rule
