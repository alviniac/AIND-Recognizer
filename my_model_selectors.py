import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def BIC_score(self, n_comp):
        model = self.base_model(n_comp)
        logL = model.score(self.X, self.lengths)    
        n_features = self.X.shape[1]
        logN = np.log(len(self.X))
        p = n_comp**2 + 2*n_comp*n_features - 1
        BIC = -2 * logL + p * logN
        return BIC


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score =  float("Inf")
            for n_comp in range(self.min_n_components, self.max_n_components+1):
                score = self.BIC_score(n_comp)
                if score < best_score:
                    best_score = score
                    best_n = n_comp
            return self.base_model(best_n)
        except:
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def DIC_score(self, n_comp):
        model = self.base_model(n_comp)
        logL = model.score(self.X, self.lengths)
        loganti = []
        for word, (seq, length) in self.hwords.items():
            if word != self.this_word:
                loganti.append(model.score(seq, length))
        DIC = logL - np.mean(loganti)
        return DIC

    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("-INF")
            for n_comp in range(self.min_n_components, self.max_n_components+1):
                score = self.DIC_score(n_comp)
                if score > best_score:
                    best_score = score
                    best_n = n_comp
            return self.base_model(best_n)
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def CV_score(self, n_comp):
        scores = []
        split_method = KFold()
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            #fit model using training set
            model = self.base_model(n_comp)
            X, lengths = combine_sequences(cv_test_idx, self.sequences)
            #score model on test set
            scores.append(model.score(X, lengths))
        CV_score = np.mean(scores)
        return CV_score
    
    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = float("-INF")
            for n_comp in range(self.min_n_components, self.max_n_components+1):
                score = self.CV_score(n_comp)
                if score > best_score:
                    best_score = score
                    best_n = n_comp
            return self.base_model(best_n)
        except:
            return self.base_model(self.n_constant)
