import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from bayesnet import BayesNetwork


# likelihood weightening
tr = pd.read_csv("input/cmc_train.csv")
ts = pd.read_csv("input/cmc_test.csv")

bn = BayesNetwork(pd.concat([tr, ts], axis=0))
bn.add_edge('wifes_age', 'wifes_edu')
bn.add_edge('wifes_age', 'wifes_rel')
bn.add_edge('n_children', 'wifes_working')
bn.add_edge('wifes_age', 'wifes_working')
bn.add_edge('husbands_occ', 'wifes_working')
bn.add_edge('sol', 'wifes_working')
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

lw = bn.likelihood_weighting(9, 100000)
print(lw.prob('media_1'))


# BayesNet
tr = pd.read_csv("input/cmc_train.csv")
ts = pd.read_csv("input/cmc_test.csv")

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

print(bn.prob('media_1'))
#y_preds = bn.predict(ts.drop('contraceptive', axis=1))
#print('accuracy={:.2f}%'.format(accuracy_score(ts['contraceptive'], y_preds) * 100))

# results
# lw(2, 1000) / lw.cond_prob('media_1') = 0.41 / bn.cond_prod('media_1') = 0.07   50s
# lw(2, 10000) / lw.cond_prob('media_1') = 0.4110801652645 / bn.cond_prod('media_1') = 0.07   50s
# lw(2, 100000) / lw.cond_prob('media_1') = 0.414491685951 / bn.cond_prod('media_1') = 0.07    8m
# lw(5, 1000) / lw.cond_prob('media_1') = 0.318756696953879 / bn.cond_prod('media_1') = 0.07   12s
# lw(7, 1000) / lw.cond_prob('media_1') = 0.182838283756438 / bn.cond_prod('media_1') = 0.07   15s
# lw(8, 1000) / lw.cond_prob('media_1') = 0.138762006575695 / bn.cond_prod('media_1') = 0.07   17s
# lw(9, 1000) / lw.cond_prob('media_1') = 0.108735385791377 / bn.cond_prod('media_1') = 0.07   17s
# lw(9, 10000) / lw.cond_prob('media_1') = 0.09545720230920 / bn.cond_prod('media_1') = 0.07   3m

# quanto mais variaveis fixas, menos tempo levar para computar e mais aproximado da probabilidade 'real'
# o numero de samples alto nao faz uma diferenca significativa no calculo da probabilidade