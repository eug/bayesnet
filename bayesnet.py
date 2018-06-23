from itertools import product
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.drawing.nx_pylab import draw_networkx
from tqdm import tqdm

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz "
                          "and either PyGraphviz or pydot")



class LikelihoodWeighting:
    WEIGHT_KEY = 'w'

    def __init__(self, weights):
        """Contruct the Likelihood Weighting given the weights table.

        Args:
            weights (DataFrame): DataFrame object containing the weights.
        """
        self.weights = weights
    

    def _get_var_val_expr(self, events):
        """Convert a list o events as (var, val) and creates a query expression.
        
        Args:
            events (list): List of events

        Returns:
            str: Returns a query expression
        """
        evts = ['({} == {})'.format(var, val) for (var, val) in events]
        return ' & '.join(evts)

    def _get_events_expr(self, events):
        """Convert a list o events as (var, event) and creates a query expression.
        
        Args:
            events (list): List of events

        Returns:
            str: Returns a query expression
        """
        evts = ["({} == '{}')".format(var, evt) for (var, evt) in events]
        return ' & '.join(evts)

    def _is_var(self, var_name):
        """Check if a variable name is a valid.

        Args:
            var_name (str): Variable name

        Returns:
            bool: Returns True when var_name is
                  a valid name, returns False otherwise.
        """
        return var_name in self.weights.columns

    def _get_var_domain(self, var_name):
        """Returns values of the variable's domain.

        Args:
            var_name (str): Variable name

        Returns:
            list: Returns a list of values of the variable's domain.
                  Raise exception if var_name is invalid.
        """
        if not self._is_var(var_name):
            msg = "Variable '{}' is not valid".format(var_name)
            raise ValueError(msg)

        vals = []
        for event in self.weights[var_name].unique():
            _, val = self._get_var_val(event)
            vals.append(val)
        return vals

    def _get_var_val(self, event):
        """Returns the variable name and value from a given event.
        
        Args:
            event (str): Event name

        Returns:
            bool: Returns a list of values of the variable's domain.
        """
        var_name = '_'.join(event.split('_')[:-1])
        var_value = int(event.split('_')[-1])
        return var_name, var_value

    def prob(self, event):
        """Compute the event probability.

        Args:
            event (str): Event name

        Returns:
            float: Returns the event probability.
        """
        var_name, var_value = self._get_var_val(event)

        if var_value not in self._get_var_domain(var_name):
            msg = "Variable value {} is not known".format(var_value)
            raise ValueError(msg)

        event_sum = self.weights[self.weights[var_name] == event][self.WEIGHT_KEY].sum()
        total_sum = self.weights[self.WEIGHT_KEY].sum()

        return event_sum / total_sum

    def cond_prob(self, hypothesis, evidencies):
        """Compute the conditional probability of the hypothesis given the evidencies.

        Args:
            hypothesis (str): Event name of the hypothesis.
            evidencies (list): List of event names representing the evidencies.

        Returns:
            float: Returns the conditional probability of the hypothesis.
        """
        hypothesis_var, _ = self._get_var_val(hypothesis)
        evidencies = list(zip(map(itemgetter(0), map(self._get_var_val, evidencies)), evidencies))
        hypothesis_expr = self._get_events_expr([(hypothesis_var, hypothesis)])
        evidence_expr = self._get_events_expr(evidencies)

        expr = '{} & {}'.format(hypothesis_expr, evidence_expr)
        event_sum = self.weights.query(expr)[self.WEIGHT_KEY].sum()
        total_sum = self.weights.query(evidence_expr)[self.WEIGHT_KEY].sum()

        return event_sum / total_sum


class BayesNetwork:
    PROB_KEY = 'p'

    def __init__(self, data):
        """Contruct the BayesNetwork object given the data table.

        Args:
            data (DataFrame): DataFrame object containing the data.
        """ 
        # Stores the training data
        self.data = data
        # Stores the parents of each vertex
        self.parent = {}
        # Stores the graph as a adjacency list
        self.graph = {}
        # Stores the probability table of a vertex
        self.table = {}
        # Stores the topological order of the graph
        self.vertices = []
        # True when all table of the graph are computed
        self.is_ready = False

    def add_edge(self, src, dst):
        """Add a graph edge.

        Args:
            src (str): Source vertex.
            dst (str): Destination vertex.
        """
        if self.is_ready:
            return

        if not self._is_var(src):
            msg = "Variable '{}' is not valid".format(src)
            raise ValueError(msg)
        
        if not self._is_var(dst):
            msg = "Variable '{}' is not valid".format(dst)
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
        """Compute the probability table of each vertex.
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
        """Find probability of hypothesis given the evidencies.

        Returns:
            float: Probability value
        """
        hypothesis_name, _ = self._get_var_val(hypothesis)

        events = [hypothesis] + evidencies
        events = [(self._get_var_val(event)[0], event) for event in events]

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


    def _is_var(self, var_name):
        """Check if a variable name is a valid.

        Args:
            var_name (str): Variable name

        Returns:
            bool: Returns True when var_name is
                  a valid name, returns False otherwise.
        """
        return var_name in self.data.columns

    def _get_var_domain(self, var_name):
        """Returns values of the variable's domain.

        Args:
            var_name (str): Variable name

        Returns:
            list: Returns a list of values of the variable's domain.
                  Raise exception if var_name is invalid.
        """
        if not self._is_var(var_name):
            msg = "Variable '{}' is not valid".format(var_name)
            raise ValueError(msg)

        return self.data[var_name].unique()

    def _get_var_val(self, event):
        """Returns the variable name and value from a given event.
        
        Args:
            event (str): Event name

        Returns:
            str: Returns a query expression
        """
        var_name = '_'.join(event.split('_')[:-1])
        var_value = int(event.split('_')[-1])
        return var_name, var_value

    def _get_events_names(self, events):
        """Create a list of event names from a list of (var, val).

        Args:
            events (list): List of events
        
        Returns:
            list: A list of event names
        """
        return ['{}_{}'.format(var, val) for (var, val) in list(events)]

    def _get_var_val_expr(self, events):
        """Convert a list o events as (var, val) and creates a query expression.
        
        Args:
            events (list): List of events

        Returns:
            str: Returns a query expression
        """
        evts = ['({} == {})'.format(var, val) for (var, val) in list(events)]
        return ' & '.join(evts)

    def _get_events_expr(self, events):
        """Convert a list o events as (var, event) and creates a query expression.
        
        Args:
            events (list): List of events

        Returns:
            str: Returns a query expression
        """
        evts = ["({} == '{}')".format(var, evt) for (var, evt) in events]
        return ' & '.join(evts)

    def prob(self, event):
        """Compute the event probability.

        Args:
            event (str): Event name

        Returns:
            float: Returns the event probability.
        """
        var_name, var_value = self._get_var_val(event)

        if var_value not in self._get_var_domain(var_name):
            msg = "Variable value '{}' is not known".format(var_value)
            raise ValueError(msg)

        df = self.prob_dist(var_name)

        return df[df[var_name] == event][self.PROB_KEY].iloc[0]

    def cond_prob(self, hypothesis, evidencies):
        """Compute the conditional probability of the hypothesis given the evidencies.

        Args:
            hypothesis (str): Event name of the hypothesis.
            evidencies (list): List of event names representing the evidencies.

        Returns:
            float: Returns the conditional probability of the hypothesis.
        """
        hypothesis_var, hypothesis_value = self._get_var_val(hypothesis)
        if hypothesis_value not in self._get_var_domain(hypothesis_var):
            msg = "Hypothesis value '{}' is not known".format(hypothesis_value)
            raise ValueError(msg)

        evidencies_vars = []
        evidencies_vars_domains = []
        for evidence in evidencies:
            evidence_var, evidence_value = self._get_var_val(evidence)
            if evidence_value not in self._get_var_domain(evidence_var):
                msg = "Evidence value '{}' is not known".format(evidence_value)
                raise ValueError(msg)
            evidencies_vars_domains.append((evidence_var, evidence_value))
            evidencies_vars.append(evidence_var)

        df = self.cond_dist(hypothesis_var, evidencies_vars)

        events = [hypothesis] + evidencies
        events = list(zip(map(itemgetter(0), map(self._get_var_val, events)), events))
        expr = self._get_events_expr(events)
        query = df.query(expr)

        if query.shape[0] > 0:
            return query[self.PROB_KEY].iloc[0]

        # No prob to return

    def prob_dist(self, var_name):
        """Compute the probability distribution of a given variable.

        Args:
            var_name (str): Variable name.

        Returns:
            DataFrame: Returns a table containing the probabilities of all events.
        """
        if not self._is_var(var_name):
            msg = "Variable name {} is not a known variable".format(var_name)
            raise ValueError(msg)

        total = 0
        table = {}
        table[var_name] = []
        table[self.PROB_KEY] = []

        for event in tqdm(self._get_var_domain(var_name), desc=var_name):
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
        """Compute the Join Distribution from a list of variables.

        Args:
            var_names (list): List of variable names.

        Returns:
            DataFrame: Returns a table containing the probabilities of all events.
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
            
            expr = self._get_var_val_expr(events)
            frequency = self.data.query(expr).shape[0] + 1
            table[self.PROB_KEY].append(frequency)
            total += frequency

        # normalization
        for i, v in enumerate(table[self.PROB_KEY]):
            table[self.PROB_KEY][i] = v / total

        return pd.DataFrame(table)

    def cond_dist(self, hypothesis, evidencies):
        """Compute the Conditional Distribution from a list of variables.

        Args:
            hypothesis (str): Event as hypothesis.
            evidencies (list): List of events as evidencies.
        Returns:
            DataFrame: Returns a table containing the probabilities of all events.
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
        for events in tqdm(all_events, desc=hypothesis):
            total = 0
            expr = self._get_var_val_expr(events)
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
        """Compute the Marginal Distribution for each variable name.

        Args:
            var_names (list): List of events as evidencies.

        Returns:
            list: Returns a list containing the marginal probability 
                  for each variable name.
        """
        subset = []
        df = self.joint_dist(var_names)
        for var_name in var_names:
            group = df.groupby(var_name)[self.PROB_KEY].sum()
            group = group.reset_index()
            subset.append(group)
        return subset

    def view(self):
        """Plot the Bayes Network.
        """
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        G = nx.DiGraph(self.graph)
        draw_networkx(G, node_size=8000, node_color='#22A7F0', pos=nx.shell_layout(G))
        plt.show()

    def predict(self, X):
        """Predict values for the dataset X .

        Args:
            X (DataFrame): Observable variables to prodict

        Returns:
            list: Returns a list of predictions
        """
        if not self.is_ready:
            self._set_ready()

        y_preds = []

        for _, x in X.iterrows():
            
            # compute probability with the network 
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

            # compute the max probability of the target variable
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
        
        return y_preds

    def _is_evidence(self, var, evidencies):
        """Check a variable is an evidencie.
        
        Args:
            var (str): Variable name
            evidencies (list): List of evidencies

        Returns:
            bool: Returns True if variable name is present
                  in evidencies, otherwise False.
        """
        for e in evidencies:
            if var in e:
                return True
        return False

    def likelihood_weighting(self, nevidencies, nsamples):
        """Sample using Likelihood Weighting of the data.

        Args:
            nevidencies (int): Number of fixed evidencies
            nsamples (int): Number of samples

        Returns:
            LikelihoodWeighting: Returns the LikelihoodWeighting object.
        """
        if not self.is_ready:
            self._set_ready()

        if nevidencies >= len(self.vertices):
            raise ValueError('The number of evidencies must be less than the number of variables')

        if nsamples <= 0:
            raise ValueError('The number of samples must be greater than zero')

        # initialize weight table
        W = {}
        W['w'] = []
        for v in self.vertices:
            W[v] = []

        # compute samples
        for _ in tqdm(range(nsamples), desc='Sampling'):

            # Select fixed evidencies
            evidencies = []
            variables = list(self.graph.keys())
            for _ in range(nevidencies):
                var = np.random.choice(variables)
                value = np.random.choice(self._get_var_domain(var))
                event = '{}_{}'.format(var, value)
                evidencies.append(event)
                W[var].append(event)
                variables.remove(var)

            w = 1

            # walk through the network
            for var in self.vertices:
                # When the variable is an evidence, it must filter all
                # variables inside its table until it finds the 
                # desired probability. Otherwise, when the variable dowe randomly choose
                # an event and add it as evidence.
                if self._is_evidence(var, evidencies):
                    
                    df = self.table[var].copy()
                    columns = list(self.table[var].columns)
                    columns.remove(self.PROB_KEY)

                    for col in columns:
                        if self._is_evidence(col, evidencies):
                            event = next(e for e in evidencies if col in e)
                            df = df[df[col] == event]
                        else:
                            value = np.random.choice(self._get_var_domain(col))
                            event = '{}_{}'.format(col, value)
                            df = df[df[col] == event]
                            evidencies.append(event)
                            W[var].append(event)

                    w *= df[self.PROB_KEY].values[0]
                else:
                    value = np.random.choice(self._get_var_domain(var))
                    event = '{}_{}'.format(var, value)
                    W[var].append(event)
                    evidencies.append(event)

            W['w'].append(w)

        return LikelihoodWeighting(pd.DataFrame(W))
