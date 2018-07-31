import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from bayesnet import BayesNetwork
import getopt


class Config:
    train_file = 'input/cmc_train.csv'
    test_file = 'input/cmc_test.csv'
    mode_predict = False
    samples = 0
    prob_event = None
    cond_events = None
    show_help = False


def parse_args(argv):
    shortopts = '_p:c:s:h'

    longopts = [
        'predict',
        'prob=',
        'cond=',
        'samples=',
        'help'
    ]

    config = Config()
    options, _ = getopt.getopt(sys.argv[1:], shortopts, longopts)

    for opt, arg in options:
        if opt == '--train':
            config.train_file = arg
        elif opt == '--test':
            config.test_file = arg
        elif opt == '--predict':
            config.mode_predict = True
        elif opt in ('-s', '--samples'):
            config.samples = int(arg)
        elif opt in ('-p', '--prob'):
            config.prob_event = arg
        elif opt in ('-c', '--cond'):
            config.cond_events = arg.split(',')
        elif opt in ('-h', '--help'):
            config.show_help = True

    return config

def print_help():
    print("""Bayes Network Demo
Usage:
    python main.py --predict
    python main.py -p wifes_age_1 -c husbands_occ_1,sol_4 -s 1000
Options:
    --predict                 Perform predictions on test dataset
    -s --samples=INT          When specified set the number of samples for Likelihood Weighting
    -p --prob=Event           Hypothesis event
    -c --cond=[<Event1>,...]  List of evidencies
    -h --help                 Print this message
    """)

if __name__ == '__main__':
    
    if len(sys.argv) <= 1:
        print('Missing arguments')
        sys.exit(1)

    config = parse_args(sys.argv[1:])

    if config.show_help:
        print_help()
        sys.exit(0)

    tr = pd.read_csv(config.train_file)
    ts = pd.read_csv(config.test_file)

    if not config.mode_predict:
        tr = pd.concat([tr, ts], axis=0)
        del ts

    bn = BayesNetwork(tr)
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

    if config.mode_predict:
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        y_true = ts['contraceptive']
        y_pred = bn.predict(ts.drop('contraceptive', axis=1))
        score = accuracy_score(y_true, y_pred) * 100
        print('Accuracy = {:.2f}%'.format(score))
        hm = sns.heatmap(confusion_matrix(y_true, y_pred), cmap='Blues', cbar=False, xticklabels=['no-use','long-term','short-term'], yticklabels=['no-use','long-term','short-term'], annot=True)
        hm.set(xlabel='Previsão', ylabel='Real')
        for item in hm.get_yticklabels():
            item.set_rotation(45)
        plt.show()
    else:
        hypothesis, evidencies = None, None
        if config.prob_event:
            hypothesis = config.prob_event

            if config.cond_events:
                evidencies = config.cond_events

            if evidencies:
                if config.samples == 0:
                    p = bn.cond_prob(hypothesis, evidencies)
                    evidencies = ','.join(config.cond_events)
                    print('P({}|{}) = {:.4f}'.format(hypothesis, evidencies, p))
                elif config.samples > 0:
                    nevidencies = len(tr.columns) - 1
                    lw = bn.likelihood_weighting(nevidencies, config.samples)
                    p = lw.cond_prob(hypothesis, evidencies)
                    evidencies = ','.join(config.cond_events)
                    print('P({}|{}) = {:.4f}'.format(hypothesis, evidencies, p))
                else:
                    print('Invalid number of samples')
                    sys.exit(1)
            else:
                if config.samples == 0:
                    p = bn.prob(hypothesis)
                    print('P({}) = {:.4f}'.format(hypothesis, p))
                elif config.samples > 0:
                    nevidencies = len(tr.columns) - 1
                    lw = bn.likelihood_weighting(nevidencies, config.samples)
                    p = lw.prob(hypothesis)
                    print('P({}) = {:.4f}'.format(hypothesis, p))
                else:
                    print('Invalid number of samples')
                    sys.exit(1)
        else:
            print('Missing --prob argument')
            sys.exit(1)
