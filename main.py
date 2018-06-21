from itertools import product
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.drawing.nx_pylab import draw_networkx
from sklearn.metrics import accuracy_score
from tqdm import tqdm

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either PyGraphviz or pydot")

class BayesNetwork:
    PROB_KEY = 'p'

    def __init__(self, data):
        self.data = data
        # it stores the parents of each vertex
        self.parent = {}
        # it stores the graph as a adjacency list
        self.graph = {}
        # it stores the probability table of a vertex
        self.table = {}
        # it stores the topological order of the graph
        self.vertices = []

        self.is_ready = False

    def add_edge(self, src, dst):
        """
        """
        if self.is_ready:
            return

        if not self._is_var(src):
            msg = "Variable {} is not valid".format(src)
            raise ValueError(msg)
        
        if not self._is_var(dst):
            msg = "Variable {} is not valid".format(dst)
            raise ValueError(msg)

        if src not in self.graph:
            self.graph[src] = []
        
        if src not in self.parent:
            self.parent[src] = []
        
        if dst not in self.parent:
            self.parent[dst] = []

        self.graph[src].append(dst)
        self.parent[dst].append(src)

    def _set_ready(self):
        """
        """
        for var_name in self.parent:
            hypothesis = var_name
            evidencies = self.parent[var_name]
            if self.parent[var_name]:
                self.table[var_name] = self.cond_dist(hypothesis, evidencies)
            else:
                self.table[var_name] = self.prob_dist(hypothesis)

        self.vertices = self._topological_sort()

        self.is_ready = True

    def _find_prob(self, hypothesis, evidencies):
        """
        """
        hypothesis_name, _ = self._get_var_name_value(hypothesis)

        events = [hypothesis] + evidencies
        events = [(self._get_var_name_value(event)[0], event) for event in events]
        expr = ' & '.join(["({} == '{}')".format(var, evt) for var, evt in events])

        return self.table[hypothesis_name].query(expr)[self.PROB_KEY].iloc[0]

    def _topological_sort(self):
        visited = []
        stack = []

        # get the first vertex that doesn't have a parent
        v = next(v for v in self.parent if not self.parent[v])
        
        # list all vertices and make sure to start in vertex that doesn't have parent
        all_vertices = [v for vals in self.graph.values() for v in vals]
        all_vertices += list(self.graph.keys())
        all_vertices = list(set(all_vertices))
        all_vertices.remove(v)
        all_vertices.insert(0, v)

        for dst in all_vertices:
            if dst not in visited:
                self._visit(dst, visited, stack)

        return stack

    def _visit(self, v, visited, stack):
        visited.append(v)

        if v not in self.graph:
            stack.append(v)
            return

        for dst in self.graph[v]:
            if dst not in visited:
                self._visit(dst, visited, stack)
        
        stack.insert(0, v)


    def _is_var(self, name):
        """
        """
        return name in self.data.columns

    def _get_var_domain(self, var_name):
        """
        """
        if self._is_var(var_name):
            return self.data[var_name].unique()

    def _get_var_name_value(self, event):
        """
        """
        var_name = '_'.join(event.split('_')[:-1])
        var_value = int(event.split('_')[-1])
        return var_name, var_value

    def _get_events_names(self, events):
        """
        """
        return ['{}_{}'.format(var_name, var_value) for (var_name, var_value) in list(events)]

    def _get_events_expr(self, events):
        """
        """
        return ' & '.join(['({} == {})'.format(var_name, var_value) for (var_name, var_value) in list(events)])

    def prob(self, event):
        """
        """
        var_name, var_value = self._get_var_name_value(event)

        if var_value not in self._get_var_domain(var_name):
            msg = "Variable value {} is not known".format(var_value)
            raise ValueError(msg)

        df = self.prob_dist(var_name)

        return df[df[var_name] == event][self.PROB_KEY].iloc[0]

    def cond_prob(self, hypothesis, evidencies):
        """
        """
        hypothesis_var_name, hypothesis_var_value = self._get_var_name_value(hypothesis)
        if hypothesis_var_value not in self._get_var_domain(hypothesis_var_name):
            return

        evidencies_vars = []
        evidencies_vars_domains = []
        for evidence in evidencies:
            evidence_var_name, evidence_var_value = self._get_var_name_value(evidence)
            if evidence_var_value not in self._get_var_domain(evidence_var_name):
                return
            evidencies_vars_domains.append((evidence_var_name, evidence_var_value))
            evidencies_vars.append(evidence_var_name)

        df = self.cond_dist(hypothesis_var_name, evidencies_vars)

        expr = []
        events = [hypothesis] + evidencies
        for event in events:
            var, _ = self._get_var_name_value(event)
            expr.append("({} == '{}')".format(var, event))
        expr = ' & '.join(expr)

        query = df.query(expr)
        if query.shape[0] > 0:
            return query[self.PROB_KEY].iloc[0]
        # TODO retornar probablidade padrao
    def prob_dist(self, var_name):
        """
        """
        if not self._is_var(var_name):
            msg = "Variable name {} is not a known variable".format(var_name)
            raise ValueError(msg)

        total = 0
        table = {}
        table[var_name] = []
        table[self.PROB_KEY] = []

        # for event in tqdm(self._get_var_domain(var_name), desc=var_name):
        for event in self._get_var_domain(var_name):
            frequency = self.data[self.data[var_name] == event].shape[0] + 1
            event_name = '{}_{}'.format(var_name, event)
            table[var_name].append(event_name)
            table[self.PROB_KEY].append(frequency)
            total += frequency

        # normalization
        for i, v in enumerate(table[self.PROB_KEY]):
            table[self.PROB_KEY][i] = v / total

        return pd.DataFrame(table)

    def joint_dist(self, var_names):
        """
        """
        if not isinstance(var_names, list):
            msg = "The parameter 'var_names' must be a list"
            raise ValueError(msg)
        
        if not all(self._is_var(var_name) for var_name in var_names):
            msg = "All items in 'var_names' must be a variable name"
            return ValueError(msg)
        
        total = 0
        table = {}
        table[self.PROB_KEY] = []
        
        vars_domains = []
        for var_name in var_names:
            table[var_name] = []
            var_value = self._get_var_domain(var_name)
            vars_domains.append(list(product([var_name], var_value)))
        
        for events in product(*vars_domains):
            for event_name in self._get_events_names(events):
                for table_key in table:
                    if table_key in event_name:
                        table[table_key].append(event_name)
                        break
            
            expr = self._get_events_expr(events)
            frequency = self.data.query(expr).shape[0] + 1
            table[self.PROB_KEY].append(frequency)
            total += frequency

        # normalization
        for i, v in enumerate(table[self.PROB_KEY]):
            table[self.PROB_KEY][i] = v / total

        return pd.DataFrame(table)

    def cond_dist(self, hypothesis, evidencies):
        """
        """
        if not self._is_var(hypothesis):
            msg = "Hypothesis {} is not a variable".format(hypothesis)
            raise ValueError(msg)
        
        if not isinstance(evidencies, list):
            msg = "The parameter 'evidencies' must be a list"
            raise ValueError(msg)
        
        if not all(self._is_var(var_name) for var_name in evidencies):
            msg = "All items in 'evidencies' must be a variable name"
            return ValueError(msg)

        table = {}
        table[self.PROB_KEY] = []

        vars_domains = []
        table[hypothesis] = []
        for var_name in evidencies:
            table[var_name] = []
            var_domain = self._get_var_domain(var_name)
            vars_domains.append(list(product([var_name], var_domain)))

        start_idx = 0
        all_events = list(product(*vars_domains))
        # for events in tqdm(all_events, desc=hypothesis):
        for events in all_events:
            total = 0
            expr = self._get_events_expr(events)
            df = self.data.query(expr)
            
            for var_value in self._get_var_domain(hypothesis):
                hypothesis_name = self._get_events_names([(hypothesis, var_value)])
                table[hypothesis].append(next(iter(hypothesis_name)))

                for event_name in self._get_events_names(events):
                    for table_key in table:
                        if table_key in event_name:
                            table[table_key].append(event_name)
                            break

                # When the we dont have a historical prob of the event
                # we set a default value: 1 / (self.data.shape[0] + 1)
                if df.shape[0] == 0:
                    table[self.PROB_KEY].append(1)
                    total += 1
                else:
                    frequency = df[df[hypothesis] == var_value].shape[0] + 1
                    total += frequency
                    table[self.PROB_KEY].append(frequency)

            # normalization
            end_idx = len(table[self.PROB_KEY])
            for i in range(start_idx, end_idx):
                table[self.PROB_KEY][i] = table[self.PROB_KEY][i] / total
            start_idx = end_idx

        return pd.DataFrame(table)

    def marginal_dist(self, var_names):
        """
        """
        subset = []
        df = self.joint_dist(var_names)
        for var_name in var_names:
            group = df.groupby(var_name)[self.PROB_KEY].sum()
            group = group.reset_index()
            subset.append(group)
        return subset

    def view(self):
        """
        """
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        G = nx.DiGraph(self.graph)
        draw_networkx(G, node_size=8000, node_color='#22A7F0', pos=nx.shell_layout(G))
        plt.show()

    def predict(self, X, y):
        """
        """
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
            for target_val in self._get_var_domain(target_var):
                evidencies = []
                hypothesis = '{}_{}'.format(target_var, target_val)
                for parent_name in self.parent[target_var]:
                    evidencies.append('{}_{}'.format(parent_name, x[parent_name]))

                proba = prod * self._find_prob(hypothesis, evidencies)

                if proba > max_proba:
                    max_proba = proba
                    y_pred = target_val

            y_preds.append(y_pred)
            # print('{:.8f}\t{}\t{}'.format(max_proba, y_pred, y))
        
        return y_preds

    def _is_evidence(self, v, evidencies):
        for e in evidencies:
            if v in e:
                return True
        return False

    def likelihood_weighting(self, nevidencies, nsamples):
        """
        """
        if not self.is_ready:
            self._set_ready()

        if nevidencies >= len(self.vertices):
            raise ValueError('The number of evidencies must be less than the number of variables')

        # initialize weight table
        W = {}
        W['w'] = []
        for v in self.vertices:
            W[v] = []

        # compute samples
        for _ in range(nsamples):

            # Select evidencies
            evidencies = []
            for _ in range(nevidencies):
                var = np.random.choice(list(self.graph.keys()))
                value = np.random.choice(self._get_var_domain(var))
                event = '{}_{}'.format(var, value)
                evidencies.append(event)

            w = 1

            # walk through the network
            for var in self.vertices:
                #print(var, evidencies)
                print(var, list(self.table[var].columns), evidencies)

                # is evidence?
                if self._is_evidence(var, evidencies):
                    # find the evidence

#                    event = next(e for e in evidencies if var in e)

                    # se eh evidencia devemos filtra na tabela da variavel
                    # quais variaveis sao conhecidas e depois aleatorizar
                    # uma valor do dominio da variavel atual para encontrar
                    # a probabilidade assosicada e entao multiplicar com o w
                    df = self.table[var].copy()
                    columns = list(self.table[var].columns)
                    columns.remove(self.PROB_KEY)
                    #print(columns)
                    for col in columns:
                        if self._is_evidence(col, evidencies):
                            event = next(e for e in evidencies if col in e)
                            #value = np.random.choice(self._get_var_domain(col))
                            #event = '{}_{}'.format(var, value)
                            df = df[df[col] == event]
                            print(event)
                        else:
                            value = np.random.choice(self._get_var_domain(col))
                            event = '{}_{}'.format(col, value)
                            df = df[df[col] == event]
                            print(event)

                    #print(df)
                    #print('-' * 20)
                    #print(self.table[var])
                    #print('#' * 200)
                    #print(self.table[var].columns, evidencies)                   
                    #self.table[var].query()
                else:
                    value = np.random.choice(self._get_var_domain(var))
                    event = '{}_{}'.format(var, value)
                    W[var].append(event)
                    evidencies.append(event)

                # has parent?
                #if self.parent[v]:
                #    pass
                #else:
                #    pass
                #self.table[v]

            print('')
        # answer 

    # likelihood_weighting(X, ['wife_age_1', 'wife_rel_0'], 1000)

tr = pd.read_csv("input/cmc_train.csv")
ts = pd.read_csv("input/cmc_test.csv")

# tr.loc[tr['contraceptive'] == 3, 'contraceptive'] = 2
# ts.loc[ts['contraceptive'] == 3, 'contraceptive'] = 2

tr = pd.concat([tr, ts], axis=0)

bn = BayesNetwork(tr)
bn.add_edge('wifes_age', 'wifes_edu')
bn.add_edge('wifes_age', 'wifes_rel')
bn.add_edge('n_children','wifes_working')
bn.add_edge('wifes_age','wifes_working')
bn.add_edge('husbands_occ','wifes_working')
bn.add_edge('sol','wifes_working')
bn.add_edge('husbands_edu', 'husbands_occ')
bn.add_edge('sol', 'n_children')
bn.add_edge('wifes_age', 'n_children')
bn.add_edge('wifes_edu', 'n_children')
bn.add_edge('media', 'n_children')
bn.add_edge('wifes_edu', 'sol')
bn.add_edge('husbands_occ', 'sol')
bn.add_edge('wifes_edu', 'media')
bn.add_edge('husbands_edu', 'media')
bn.add_edge('wifes_rel', 'media')
bn.add_edge('wifes_age', 'contraceptive')
bn.add_edge('wifes_edu', 'contraceptive')
bn.add_edge('n_children', 'contraceptive')
bn.add_edge('wifes_working', 'contraceptive')


#bn.view()

#X, y = df.drop('contraceptive', axis=1), df['contraceptive']
bn.likelihood_weighting(2, 1000)


# y_preds = bn.predict(ts.drop('contraceptive', axis=1), ts['contraceptive'])
# print('accuracy={:.2f}%'.format(accuracy_score(ts['contraceptive'], y_preds) * 100))
