from statistics import mean
from collections import Counter
import itertools
import random
import pandas as pd


def minLength(rules):
    return len(min([r[:-1] for r in rules], key=len))


def maxLength(rules):
    return len(max([r[:-1] for r in rules], key=len))


def avgLength(rules):
    return mean([len(r[:-1]) for r in rules])


def minSupport(rules, df):
    return min([support(df, r) for r in rules])


def maxSupport(rules, df):
    return max([support(df, r) for r in rules])


def avgSupport(rules, df):
    return mean([support(df, r) for r in rules])


def checkMinClassValue(df):
    classVals = df[list(df.columns)[-1]].tolist()
    cnt = Counter(classVals)
    mincnt = min(cnt.values())
    minval = next(n for n in reversed(classVals) if cnt[n] == mincnt)

    return classVals.count(minval)


def checkMostCommonClassValueOfRules(rules):
    grouped_rules = [list(g)
                     for i, g in itertools.groupby(rules, lambda x: x[-1])]
    rule_max = [len(k) for k in grouped_rules]
    rule_ids = [i for i, j in enumerate(rule_max) if j == max(rule_max)]
    voteMaxLen = [grouped_rules[v] for v in rule_ids]

    return voteMaxLen[0][0]


def intersection(l_rules, rows):
    r = [list(i.keys())[0] for i in l_rules]
    return list(set(r[:-1]) & set(rows[:-1]))


def common_member(df_attr: list, rule: list, with_class=True):
    df_attr = list(set(df_attr[:-1]))
    rule_contd = list(set([list(i.keys())[0] for i in rule[:-1]]))

    return all(x in df_attr for x in rule_contd)


def get_matched_rule_in_data_frame(df, rule, with_class=True):
    dff = df.astype(str)
    if with_class:
        attr = list(df.columns)
        rule[-1][attr[-1]] = rule[-1].pop(list(rule[-1].keys())[0])
    else:
        attr = list(df.columns)[:-1]
        rule = rule[:-1]

    if common_member(df.columns, rule, with_class=True):
        for dr in attr:
            for r in rule:
                if dr == list(r.keys())[0]:
                    dff = dff.loc[dff[dr].isin([str(r[dr])])]
        return dff
    else:
        return pd.DataFrame()


def support(df, rule):
    return get_matched_rule_in_data_frame(df, rule, True).shape[0]


def set_of_support(df, rules):
    decSupport = []
    for rule in rules:
        decSupport.append(support(df, rule))

    return decSupport


def equal_ignore_order(a, b):
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched


def getMaxRuleSupport(votes: list):
    chosenRule = max(votes, key=lambda x: x['support'])['rule']
    chosenSupport = max(votes, key=lambda x: x['support'])['support']

    return chosenRule, chosenSupport


def votedRule(votes: list):
    rule_max = [len(k['rule']) for k in votes]
    rule_ids = [i for i, j in enumerate(rule_max) if j == max(
        rule_max)]  # position of rules with highest length
    voteMaxLen = [votes[v] for v in rule_ids]  # rules with highest length
    chosenRule, chosenSupport = getMaxRuleSupport(voteMaxLen)

    return chosenRule, chosenSupport


def standardVoting(votes: list):
    candidate_rules = [k['rule'] for k in votes]
    grouped_rules = [list(g) for i, g in itertools.groupby(
        candidate_rules, lambda x: x[-1])]

    rule_max = [len(k) for k in grouped_rules]
    rule_ids = [i for i, j in enumerate(rule_max) if j == max(rule_max)]
    voteMaxLen = [grouped_rules[v] for v in rule_ids]

    maxDecRules = [{'rule': j, 'support': votes[candidate_rules.index(j)]['support']}
                   for i in voteMaxLen for j in i]

    return getMaxRuleSupport(maxDecRules)


def decisionListVoting(votes: list):
    return votes[0]['rule'], votes[0]['support']


def score(testdata, rules, dsupport):
    c = 0
    listofvote = []
    listofsupport = []
    if len(rules) == 0:
        return 'List of rules is empty!!!'
    for idx in list(testdata.index):
        votes = []
        supportV = -1
        for idr, r in enumerate(rules):
            predict = []
            for i in intersection(r, testdata.columns):
                predict.append({i: testdata.loc[idx, i]})
            if [j for j in predict if j not in r[:-1]] == []:
                votes.append({'rule': r, 'support': dsupport[idr]})
        if len(votes) > 0:
            vote, supportV = standardVoting(votes)
        else:
            # randomly choose a rule
            randindex = random.randint(0, len(rules))
            vote = rules[randindex-1]
            supportV = dsupport[randindex-1]
#             vote = checkMostCommonClassValueOfRules(rules)
#             supportV = dsupport[rules.index(vote)]

        if vote[-1][list(testdata.columns)[-1]] == list(testdata.loc[idx])[-1]:
            c += 1
        listofvote.append(vote)
        listofsupport.append(supportV)

    return c/len(list(testdata.index)), listofvote, listofsupport


def clean_rules(files: list):
    l = []
    for f in range(len(files)):
        rules = []
        for i in range(len(files[f])):
            if files[f][i]["feature_name"] != "":
                rules.append(files[f][i])
        l.append(rules)
    return l

# get all decision nodes from the json decision rules


def set_of_tree_decisions(files: list):
    l = []
    rules = clean_rules(files)
    for i in range(len(rules)):
        t = []
        for j in range(len(rules[i])):
            if rules[i][j]["return_statement"] == 1:
                t.append([rules[i][j]])
        l.append(t)
    return l

# get the set of decision rules from json


def get_rules(files: list):
    decisionNodes = set_of_tree_decisions(files)
    rules = clean_rules(files)
    for i in range(len(decisionNodes)):
        for j in range(len(decisionNodes[i])):
            while decisionNodes[i][j][-1]["current_level"] != 1:
                for k in range(len(rules[i])):
                    if decisionNodes[i][j][-1]["parents"] == rules[i][k]["leaf_id"]:
                        decisionNodes[i][j].append(rules[i][k])
    return decisionNodes

# format the rules in the of list of dict eg. [{'f1': 0 },...,{'d': value}]


def format_rules(files: list):
    out = []

    rules = get_rules(files)

    for i in range(len(rules)):
        tmp = []
        value = ""
        for j in range(len(rules[i])):
            inner = []
            for k in range(len(rules[i][j])):
                ruleStrip = rules[i][j][k]["rule"].replace(
                    ' ', '').replace(':', '')
                if '<=' in ruleStrip:
                    value = ruleStrip.split('<=')[-1]
                elif '==' in ruleStrip:
                    value = ruleStrip.split('==')[-1].replace("'", "")
                elif '>' in ruleStrip:
                    value = ruleStrip.split('>')[-1]
                else:
                    value = rules[i][j][k]["rule"]
                if rules[i][j][k]["return_statement"] == 0:
                    inner.append({rules[i][j][k]["feature_name"]: value})
                else:
                    inner.append(
                        {"d": value.replace("return", "").replace("'", "").strip()})
            tmp.append(inner)
        for t in range(len(tmp)):
            tmp[t].reverse()
        out.append(tmp)
    return out


def format_rule(rule):
    formatted_rule = ', '.join(
        [f"{k}: {v}" for item in rule[:-1] for k, v in item.items()])
    formatted_rule += f" -> {'d'}: {rule[-1][list(rule[-1].keys())[0]]}"

    return formatted_rule
