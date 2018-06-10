from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def is_var(data, name):
    return name in data.columns

def get_var_domain(data, var_name):
    if is_var(data, var_name):
        return data[var_name].unique()

def get_var_name_value(event):
    var_name = '_'.join(event.split('_')[:-1])
    var_value = int(event.split('_')[-1])
    return var_name, var_value

def get_events_names(events):
    return ['{}_{}'.format(var_name, var_value) for (var_name, var_value) in list(events)]

def get_events_expr(events):
    return ' & '.join(['({} == {})'.format(var_name, var_value) for (var_name, var_value) in list(events)])

def prob(data, event):
    prob_key = 'p'
    var_name, var_value = get_var_name_value(event)

    if var_value not in get_var_domain(data, var_name):
        return

    df = prob_dist(data, var_name)

    return df[df[var_name] == event][prob_key].iloc[0]

# prob(tr, 'wifes_rel_1')

def prob_dist(data, var_name):
    if not is_var(data, var_name):
        return
    
    prob_key = 'p'
    total = 1
    table = {}
    table[prob_key] = []
    table[var_name] = []
        
    for event in tqdm(get_var_domain(data, var_name), desc=var_name):
        frequency = data[data[var_name] == event].shape[0]
        event_name = '{}_{}'.format(var_name, event)
        table[var_name].append(event_name)
        table[prob_key].append(frequency)
        total += frequency
    
    for i, v in enumerate(table[prob_key]):
        table[prob_key][i] = v / total
    
    return pd.DataFrame(table)

def joint_dist(data, var_names):
    if not isinstance(var_names, list):
        return
    
    if not all(is_var(data, var_name) for var_name in var_names):
        return
    
    prob_key = 'p'
    total = 1
    table = {}
    table[prob_key] = []
    
    vars_domains = []
    for var_name in var_names:
        table[var_name] = []
        var_domain = get_var_domain(data, var_name)
        vars_domains.append(list(product([var_name], var_domain)))
    
    for events in product(*vars_domains):
        for event_name in get_events_names(events):
            for table_key in table:
                if table_key in event_name:
                    table[table_key].append(event_name)
                    break
        
        expr = get_events_expr(events)
        frequency = data.query(expr).shape[0]
        table[prob_key].append(frequency)
        total += frequency

    for i, v in enumerate(table[prob_key]):
        table[prob_key][i] = v / total
    
    return pd.DataFrame(table)

# joint_dist(tr, ['wifes_age', 'sol', 'wifes_rel', 'n_children'])

def cond_dist(data, hypothesis, evidencies):
    if not is_var(data, hypothesis):
        return
    
    if not isinstance(evidencies, list):
        return
    
    if not all(is_var(data, var_name) for var_name in evidencies):
        return

    prob_key = 'p'
    table = {}
    table[prob_key] = []

    vars_domains = []
    table[hypothesis] = []
    for var_name in evidencies:
        table[var_name] = []
        var_domain = get_var_domain(data, var_name)
        vars_domains.append(list(product([var_name], var_domain)))

    all_events = list(product(*vars_domains))
    for events in tqdm(all_events, desc=hypothesis):
        expr = get_events_expr(events)
        df = data.query(expr)

        for var_domain in get_var_domain(data, hypothesis):
            hypothesis_name = get_events_names([(hypothesis, var_domain)])
            table[hypothesis].append(next(iter(hypothesis_name)))
            for event_name in get_events_names(events):
                for table_key in table:
                    if table_key in event_name:
                        table[table_key].append(event_name)
                        break

            frequency = df[df[hypothesis] == var_domain].shape[0]
            total = df.shape[0] + 1
            table[prob_key].append(frequency / total)

    return pd.DataFrame(table)

# cond_dist(tr, 'wifes_age', ['wifes_working', 'wifes_rel'])

def marginal_dist(data, var_names):
    subset = []
    prob_key = 'p'
    df = joint_dist(data, var_names)
    for var_name in var_names:
        subset.append(df.groupby(var_name)[prob_key].sum().reset_index())
    return subset

# marginal_dist(tr, ['husbands_edu', 'husbands_occ'])

def cond_prob(data, hypothesis, evidencies):
    hypothesis_var_name, hypothesis_var_domain = get_var_name_value(hypothesis)
    if not is_domain_of(data, hypothesis_var_name, hypothesis_var_domain):
        return
    
    prob_key = 'p'
    evidencies_vars = []
    evidencies_vars_domains = []
    for evidence in evidencies:
        # TODO: CHECK FOR EVIDENCIES VAR NAMES
        evidence_var_name, evidence_var_domain = get_var_name_value(evidence)
        evidencies_vars_domains.append((evidence_var_name, evidence_var_domain))
        evidencies_vars.append(evidence_var_name)

    df = cond_dist(data, hypothesis_var_name, evidencies_vars)

    expr = []
    events = [hypothesis] + evidencies
    for event in events:
        var, _ = get_var_name_value(event)
        expr.append("({} == '{}')".format(var, event))
    expr = ' & '.join(expr)

    return df.query(expr)[prob_key].iloc[0]
    
# cond_prob(tr, 'wifes_working_1', ['wifes_rel_1', 'wifes_edu_1'])

class BayesNetwork:
    def __init__(self, data):
        self.data = data
        self.parent = {}
        self.graph = {}
        self.table = {}
        self.is_ready = False

    def add_edge(self, src, dst):
        if self.is_ready:
            return

        if not is_var(self.data, src) or\
           not is_var(self.data, dst):
            return

        if src not in self.graph:
            self.graph[src] = []
        
        if src not in self.parent:
            self.parent[src] = []
        
        if dst not in self.parent:
            self.parent[dst] = []

        self.graph[src].append(dst)
        self.parent[dst].append(src)

    def _set_ready(self):
        for var_name in self.parent:
            hypothesis = var_name
            evidencies = self.parent[var_name]
            if self.parent[var_name]:
                self.table[var_name] = cond_dist(self.data, hypothesis, evidencies)
            else:
                self.table[var_name] = prob_dist(self.data, hypothesis)
        self.is_ready = True

    def _find_prob(self, hypothesis, evidencies):
        prob_key = 'p'
        hypothesis_name, _ = get_var_name_value(hypothesis)

        events = [hypothesis] + evidencies
        events = [(get_var_name_value(event)[0], event) for event in events]
        expr = ' & '.join(["({} == '{}')".format(var, evt) for var, evt in events])

        return self.table[hypothesis_name].query(expr)[prob_key].iloc[0]


    def view(self):
        pass

    def predict(self, X, y):
        if not self.is_ready:
            self._set_ready()

        y_preds = []

        for (_, x), y in zip(X.iterrows(), y):
            
            prod, target_var = 1, None

            for var_name in self.parent:
                if var_name not in x.keys():
                    target_var = var_name
                    continue

                evidencies = []
                hypothesis = '{}_{}'.format(var_name, x[var_name])
                for parent_name in self.parent[var_name]:
                    evidencies.append('{}_{}'.format(parent_name, x[parent_name]))

                prod *= self._find_prob(hypothesis, evidencies)

            y_pred, max_proba = -1, -1
            for target_val in get_var_domain(self.data, target_var):
                evidencies = []
                hypothesis = '{}_{}'.format(target_var, target_val)
                for parent_name in self.parent[target_var]:
                    evidencies.append('{}_{}'.format(parent_name, x[parent_name]))

                proba = prod * self._find_prob(hypothesis, evidencies)

                if proba > max_proba:
                    max_proba = proba
                    y_pred = target_val

            y_preds.append(y_pred)
            print('{:.8f}\t{}\t{}'.format(max_proba, y_pred, y))
        
        return y_preds
        


tr = pd.read_csv("input/cmc_train.csv")
ts = pd.read_csv("input/cmc_test.csv")

# tr.loc[tr['contraceptive'] == 3, 'contraceptive'] = 2
# ts.loc[ts['contraceptive'] == 3, 'contraceptive'] = 2

bn = BayesNetwork(tr)
# bn.add_edge('husbands_edu', 'husbands_occ')
# bn.add_edge('husbands_occ', 'sol')
# bn.add_edge('wifes_age', 'wifes_edu')
# bn.add_edge('wifes_edu', 'media')
# bn.add_edge('wifes_edu', 'sol')
# bn.add_edge('wifes_age', 'n_children')
# bn.add_edge('wifes_rel', 'wifes_working')
# bn.add_edge('wifes_working', 'n_children')
# bn.add_edge('n_children', 'contraceptive')
# bn.add_edge('media', 'contraceptive')
# bn.add_edge('sol', 'contraceptive')


# bn.add_edge('husbands_edu', 'husbands_occ')
# bn.add_edge('husbands_occ', 'sol')
# bn.add_edge('wifes_age', 'n_children')
# bn.add_edge('media', 'n_children')
# bn.add_edge('wifes_rel', 'n_children')
# bn.add_edge('wifes_edu', 'sol')
# bn.add_edge('wifes_edu', 'wifes_working')
# bn.add_edge('n_children', 'contraceptive')
# bn.add_edge('wifes_working', 'contraceptive')
# bn.add_edge('sol', 'contraceptive')

# bn.add_edge('husbands_edu', 'contraceptive')
# bn.add_edge('husbands_occ', 'contraceptive')
# bn.add_edge('wifes_age', 'contraceptive')
# bn.add_edge('wifes_edu', 'contraceptive')
# bn.add_edge('wifes_rel', 'contraceptive')
# bn.add_edge('wifes_working', 'contraceptive')
# bn.add_edge('n_children', 'contraceptive')
# bn.add_edge('media', 'contraceptive')
# bn.add_edge('sol', 'contraceptive')

bn.add_edge('wifes_edu', 'wifes_age')
bn.add_edge('wifes_working', 'wifes_age')
bn.add_edge('husbands_occ', 'wifes_age')
bn.add_edge('n_children', 'wifes_age')
bn.add_edge('wifes_age', 'wifes_rel')
bn.add_edge('media', 'wifes_rel')
bn.add_edge('n_children','wifes_working')
bn.add_edge('wifes_age','wifes_working')
bn.add_edge('hunbands_occ','wifes_working')
bn.add_edge('sol','wifes_working')
bn.add_edge('husbands_edu', 'husbands_occ')
bn.add_edge('sol', 'n_children')
bn.add_edge('wifes_age', 'n_children')
bn.add_edge('wifes_edu', 'n_children')
bn.add_edge('media', 'n_children')
bn.add_edge('husbands_edu', 'n_children')
bn.add_edge('husbands_edu', 'sol')
bn.add_edge('wifes_edu', 'sol')
bn.add_edge('husbands_occ', 'sol')
bn.add_edge('wifes_edu', 'media')
bn.add_edge('husbands_edu', 'media')
bn.add_edge('wifes_rel', 'media')
bn.add_edge('n_children', 'media')
bn.add_edge('husbands_edu', 'contraceptive')
bn.add_edge('wifes_age', 'contraceptive')
bn.add_edge('wifes_edu', 'contraceptive')
bn.add_edge('wifes_rel', 'contraceptive')
bn.add_edge('n_children', 'contraceptive')
bn.add_edge('media', 'contraceptive')
bn.add_edge('sol', 'contraceptive')

y_preds = bn.predict(ts.drop('contraceptive', axis=1), ts['contraceptive'])
print('accuracy={:.2f}%'.format(accuracy_score(ts['contraceptive'], y_preds) * 100))