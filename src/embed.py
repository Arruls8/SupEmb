"""
Loads the input matrices created by the createMatrices.py module and
implements the supervised embedding (SupEmb) method.
"""

__author__ = 'Danushka Bollegala'
__license__ = 'BSD'
__date__ = "18/04/2014"


import numpy
import scipy.sparse
from scipy.io import mmwrite, mmread

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import sys
import time

from sparsesvd import sparsesvd

# catch warnings as errors
import warnings
warnings.filterwarnings("always")

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import ipdb


# debug info file
debug_file = None

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logger.info('func:%r took: %2.4f sec' %  (f.__name__, te-ts))
        return result

    return timed


class SupEmb():


    def __init__(self, Ua=None, Ub=None, A=None, B=None, XlA_pos=None, XlA_neg=None, XuA=None, XuB=None):

        if Ua is not None and Ub is not None and A is not None and B is not None \
            and XlA_pos is not None and XlA_neg is not None  and XuA is not None and XuB is not None:
            self.Ua = Ua 
            logger.info("Ua %d x %d" % Ua.shape)
            self.Ub = Ub
            logger.info("Ub %d x %d" % Ub.shape)
            self.A = A
            logger.info("A %d x %d" % A.shape)
            self.B = B
            logger.info("B %d x %d" % B.shape)
            self.XlA_pos = XlA_pos
            logger.info("XlA_pos %d x %d" % XlA_pos.shape)
            self.XlA_neg = XlA_neg
            logger.info("XlA_neg %d x %d" % XlA_neg.shape)
            self.XuA = XuA
            logger.info("XuA %d x %d" % XuA.shape)
            self.XuB = XuB
            logger.info("XuB %d x %d" % XuB.shape)
            assert(self.Ua.shape[0] == self.Ub.shape[0])
            self.M = self.Ua.shape[0]
            assert(self.Ub.shape[1] == self.B.shape[1])
            assert(self.Ua.shape[1] == self.A.shape[1])
            assert(self.XlA_pos.shape[1] == self.XlA_neg.shape[1])
            assert(self.Ua.shape[0] + self.A.shape[0] == self.XuA.shape[1])
            assert(self.Ub.shape[0] + self.B.shape[0] == self.XuB.shape[1])
            self.d = self.Ua.shape[1]
            self.h = self.Ub.shape[1]

        # parameters of the model.
        self.w1 = 1.0  # weight for Rule 1.
        self.w2 = 1.0 # weight for Rule 2.
        self.w3 = 1.0 # weight for Rule 3.
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        self.k2 = 5 # k for k-NN when computing W2
        self.k3 = 5 # k for k-NN when computing W3
        self.k3_bar = 5 # k for k-NN when computing W3_bar
        self.dims = 50 # number of latent dimensions for the embedding
        self.mu = 1.0 # how much emphasise should we make on the projected features.
        self.debug = True
        pass


    @timeit
    def get_W1(self, M):
        """
        Create a 2Mx2M matrix of the form [O I; I O]
        where I is MxM unit matrix and O is MxM zero matrix. 
        """
        O = numpy.zeros((M, M), dtype=float)
        I = numpy.eye(M, dtype=float)
        top = numpy.concatenate((O, I), axis=1)
        bottom = numpy.concatenate((I, O), axis=1)
        W1 = numpy.concatenate((top, bottom), axis=0)
        return W1


    @timeit    
    def get_W2(self):
        # Create the matrix for all labeled instances.
        self.XlA = numpy.concatenate((self.XlA_pos, self.XlA_neg), axis=0)
        self.pos_n = self.XlA_pos.shape[0]
        self.neg_n = self.XlA_neg.shape[0]
        n = self.pos_n + self.neg_n
        self.Y = numpy.concatenate((numpy.ones(self.pos_n), -1 * numpy.ones(self.neg_n)))
        S, neighbours = self.get_kNNs(self.XlA, self.k2)
        W2 = numpy.zeros((n, n), dtype=float)
        for i in range(0, n):
            for j in range(0, n):
                if (j in neighbours[i]) and (i in neighbours[j]):
                    # i and j are undirected nearest neighbours. 
                    if self.Y[i] == self.Y[j]:
                        W2[i,j] = -self.lambda_1 * S[i,j]
                    else:
                        W2[i,j] = S[i,j]
                
        return W2


    @timeit    
    def get_W3(self, X, k):
        """
        Rule 3. Source domain unlabeled documents neighbourhood constraint. 
        """
        S, neighbours = self.get_kNNs(X, k)
        n = X.shape[0]
        W3 = numpy.zeros((n, n), dtype=float)
        for i in range(0, n):
            for j in range(0, n):
                # i and j are undirected nearest neighbours. 
                W3[i,j] = S[i,j]
        return W3


    @timeit
    def get_kNNs(self, A, k):
        """
        A is a matrix where each row represents a document over a set 
        of features represented in columns. We will first normalise 
        the rows in A to unit L2 length and then measure the cosine 
        similarity between all pairs of rows (documents). We will then 
        select the most similar k documents for each document in A.
        We will then return a dictionary of the format
        neighbours[rowid] = numpy.array of neighbouring row ids
        Note that the neighbours within the list are not sorted according
        to their similarity scores to the base document. 

        Args:
            A (numpy.array): document vs. feature matrix. 
            k (integer): nearest neighbours

        Returns:
            S (numpy.array): similarity matrix
            neighbours (dictionary): mapping between row ids and 
                the list of neighbour row ids. 
        """
        # normalise rows in A to unit L2 length.
        #normA = preprocessing.normalize(A, norm='l2', copy=True)
        normA = A
        # Compute similarity
        S = numpy.dot(normA, normA.T)
        N = numpy.argsort(S)
        neighbours = {}
        for i in range(S.shape[0]):
            neighbours[i] = N[i,:][-k:]
        return (S, neighbours)


    @timeit  
    def get_Laplacian(self, W):
        """
        Compute the Laplacian of matrix W. 
        """
        # axix=0 column totals, axis=1 row totals.
        s = numpy.array(numpy.sum(W, axis=1))
        D = numpy.diag(s)
        L = W - D
        return L


    def perturbate(self, s):
        """
        Replace zero valued elements in s by the minimum value 
        of s. This must be done to avoid zero division. 
        """
        v = numpy.mean(s)
        zeros = numpy.where(s == 0)
        for i in zeros:
            s[i] = v
        logging.warning("Perturbations = %d" % len(zeros))
        return s


    @timeit
    def get_Dinv(self, W):
        """
        Compute the D^{-1} matrix of W. 
        The (i,i) diagonal element of this diagonal matrix is set
        to 1/sum_of_i_th_row_in_W. 
        """
        s = numpy.array(numpy.sum(W, axis=1))
        if numpy.count_nonzero(s) > 0:
            # zero sum documents exist! 
            s = self.perturbate(s)
        Dinv = numpy.diag(1.0 / s)
        return Dinv


    @timeit
    def get_U1(self):
        """
        U1 = [[Ua O], [0 Ub]]
        """
        Oh = numpy.zeros((self.M, self.h), dtype=float)
        top = numpy.concatenate((self.Ua, Oh), axis=1)
        Od = numpy.zeros((self.M, self.d), dtype=float)
        bottom = numpy.concatenate((Od, self.Ub), axis=1)
        U1 = numpy.concatenate((top, bottom), axis=0)
        return U1

    @timeit
    def get_embedding(self):
        """
        Compute the embedding matrix P. 
        """
        # Compute the components for Rule 1.
        global debug_file
        logger.info("Rule1")
        W1 = self.get_W1(self.M)
        L1 = self.get_Laplacian(W1)
        U1 = self.get_U1()
        logger.info("Computing U1.T * L(W1) * U1")
        Qright = numpy.dot(numpy.dot(U1.T, L1), U1)

        # Compute the components for Rule 2.
        logger.info("Rule2")
        logger.info("Computing W2")
        W2 = self.get_W2()
        L2 = self.get_Laplacian(W2)
        D2 = self.get_Dinv(self.XlA)
        totA = numpy.concatenate((self.Ua, self.A), axis=0)
        logger.info("Computing F2")
        part1 = numpy.dot(numpy.dot(D2, self.XlA), totA)
        F2 = numpy.dot(numpy.dot(part1.T, L2), part1)
        
        # Compute the components for Rule 3.
        logger.info("Rule3")
        logger.info("Computing W3")
        #ipdb.set_trace()
        W3 = self.get_W3(self.XuA, self.k3)
        L3 = self.get_Laplacian(W3)
        D3 = self.get_Dinv(self.XuA)
        logger.info("Computing F3")
        part2 = numpy.dot(numpy.dot(D3, self.XuA), totA)
        F3 = numpy.dot(numpy.dot(part2.T, L3), part2)

        logger.info("Computing W3_bar")
        W3_bar = self.get_W3(self.XuB, self.k3_bar)
        L3_bar = self.get_Laplacian(W3_bar)
        D3_bar = self.get_Dinv(self.XuB)
        totB = numpy.concatenate((self.Ub, self.B), axis=0)
        logger.info("Computing F3_bar")
        part3 = numpy.dot(numpy.dot(D3_bar, self.XuB), totB)
        F3_bar = numpy.dot(numpy.dot(part3.T, L3_bar), part3)

        logger.info("Computing Q")
        Q11 = (self.w2 * F2) - (self.w3 * F3)
        Q12 = numpy.zeros((self.d, self.h), dtype=float)
        Q22 = -self.w3 * self.lambda_2 * F3_bar
        Qtop = numpy.concatenate((Q11, Q12), axis=1)
        Qbottom = numpy.concatenate((Q12.T, Q22), axis=1)
        Qleft = numpy.concatenate((Qtop, Qbottom), axis=0)
        Q = Qleft - (self.w1 * Qright) 

        # Compute the values of Q1, Q2, and Q3 for debugging purposes.
        if self.debug:
            # Compute Q1.
            O1 = numpy.trace(Qright)
            # Compute Q2.
            Q2_top = numpy.concatenate((F2, numpy.zeros((self.d, self.h))), axis=1)
            Q2_bottom = numpy.concatenate((numpy.zeros((self.h, self.d)), numpy.zeros((self.h, self.h))), axis=1)
            Q2 = numpy.concatenate((Q2_top, Q2_bottom), axis=0)
            O2 = numpy.trace(Q2)
            # Compute Q3.
            Q3_top = numpy.concatenate((F3, numpy.zeros((self.d, self.h))), axis=1)
            Q3_bottom = numpy.concatenate((numpy.zeros((self.h, self.d)), self.lambda_2 * F3_bar), axis=1)
            Q3 = numpy.concatenate((Q3_top, Q3_bottom), axis=0)
            O3 = numpy.trace(Q3)
            debug_file.write("%s, %s, %s\n" % (str(O1), str(O2), str(O3)))
            debug_file.flush()
            logger.info("O1 = %s" % str(O1))
            logger.info("O2 = %s" % str(O2))
            logger.info("O3 = %s" % str(O3))
        return Q


    @timeit
    def get_projection(self, Q):
        """
        Compute the projection matrices Pa and Pb for the 
        source and the target domains. 
        """
        logger.info("Dimensionality of Q: %d x %d" % Q.shape)
        matsize = numpy.sum(numpy.abs(Q))
        logger.info("Sum of Q = %f" % matsize)
        
        if matsize == 0:
            # cannot use sparsesvd to SVD a zero matrix. Use numpy.linalg.svd
            u, s, v = numpy.linalg.svd(Q)
            u = u[:self.dims, :]
        else:
            u, s, v = sparsesvd(scipy.sparse.csc_matrix(Q), self.dims)
        
        #I = numpy.dot(u, v.T)
        #numpy.testing.assert_array_almost_equal_nulp(I, numpy.eye(self.dims))
        logger.info("Source feature space dimensions = %d" % self.d)
        logger.info("Target feature space dimensions = %d" % self.h)
        Pa = u[:, :self.d]
        Pb = u[:, self.d:]
        logger.info("Pa %d x %d" % Pa.shape)
        logger.info("Pb %d x %d" % Pb.shape)
        return Pa, Pb


    def project_instances(self, X, U, A, P):
        """
        Project train or test instances which are in rows of X 
        according to the projection matrix P.
        """
        Y = numpy.dot(X.T, self.get_Dinv(X))
        left = numpy.dot(U, P.T)
        right = numpy.dot(A, P.T)
        Z = numpy.concatenate((left.T, right.T), axis=1)
        Z = numpy.dot(Z, Y).T
        logger.info("Dimensionality of the Projected Matrix %d x %d" % Z.shape)
        return Z


    def load_pivots(self, pivots_filename):
        pivots = []
        pivots_file = open(pivots_filename)
        for line in pivots_file:
            pivots.append(line.strip().split()[1])
        pivots_file.close()
        return pivots


    def load_feature_space(self, fname):
        """
        Loads the nonpivots for the domain. 
        """
        D = []
        F = open(fname)
        for line in F:
            D.append(line.split('\t')[1].strip())
        F.close()
        return D


    # def get_word_feature_index(self, base_path):
    #     """
    #     Read the source_feats and target_feats files in the base_path directory
    #     and compute their union. This is important when we concatenate 
    #     projected features with the original features because the ids of the original
    #     features must be consistent between the source and the target domains. 
    #     These features are used to represent words (ie. pivots and non-pivots).
    #     """
    #     source_feats = []
    #     target_feats = []
    #     feat_index = []
    #     source_feat_file = open("%s/source_feats" % base_path)
    #     for line in source_feat_file:
    #         feat = line.split('\t')[1].strip()
    #         source_feats.append(feat)
    #         if feat not in feat_index:
    #             feat_index.append(feat)
    #     source_feat_file.close()
    #     target_feat_file = open("%s/target_feats" % base_path)
    #     for line in target_feat_file:
    #         feat = line.split('\t')[1].strip()
    #         target_feats.append(feat)
    #         if feat not in feat_index:
    #             feat_index.append(feat)
    #     target_feat_file.close()
    #     return (source_feats, target_feats, feat_index)


    def concatenate_original_projected(self, Z, X, domain_feats, feat_index):
        """
        Concatenates original and the projected vectors. 
        Z is the projected features. 
        X is the original features. 
        """
        no_of_docs = X.shape[0]
        M = numpy.zeros((no_of_docs, len(feat_index)), dtype=float)
        for i in range(0, no_of_docs):
            for j in range(0, X.shape[1]):
                if X[i,j] != 0:
                    feat_name = domain_feats[j]
                    ind = feat_index.index(feat_name)
                    M[i, ind] = X[i,j]
        

        return numpy.concatenate((self.mu * Z, M), axis=1)
        #return M


    def save_embedding(self, filename, Q):
        """
        Save the embedding Q to a disk file.
        """
        mmwrite(filename, Q)
        pass


    def load_embedding(self, filename):
        """
        Load the embedding from the filename.
        """
        return mmread(filename)


    def check_symmetry(self, Q):
        """
        Checks whether Q is symmetric.
        """
        numpy.testing.assert_array_almost_equal_nulp(Q, Q.T)
        pass
    pass


def load_matrix(fname):
    """
    Loads the matrix from the matrix market format. 
    """
    M = mmread(fname)
    return M.toarray()


@timeit
def train_logistic(pos_train, neg_train):
    """
    Train a binary logistic regression classifier 
    using the positive and negative train instances.
    """
    LR = LogisticRegression(penalty='l2', C=0.1)
    pos_n = pos_train.shape[0]
    neg_n = neg_train.shape[0]
    y =  numpy.concatenate((numpy.ones(pos_n), -1 * numpy.ones(neg_n)))
    X = numpy.concatenate((pos_train, neg_train), axis=0)
    logger.info("Train Positive Feature Dimensions = %d" % pos_train.shape[1])
    logger.info("Train Negative Feature Dimensions = %d" % neg_train.shape[1])
    LR.fit(X, y)
    logger.info("Train accuracy = %f " % LR.score(X, y))
    return LR


def test_logistic(pos_test, neg_test, model):
    """
    Classify the test instances using the trained model.
    """
    pos_n = pos_test.shape[0]
    neg_n = neg_test.shape[0]
    y =  numpy.concatenate((numpy.ones(pos_n), -1 * numpy.ones(neg_n)))
    X = numpy.concatenate((pos_test, neg_test), axis=0)
    logger.info("Test Positive Feature Dimensions = %d" % pos_test.shape[1])
    logger.info("Test Negative Feature Dimensions = %d" % neg_test.shape[1])
    accuracy = model.score(X, y)
    logger.info("Test accuracy = %f" % accuracy)
    return accuracy


def process(source_domain, target_domain, w1=1.0, w2=1.0, w3=1.0, lambda_1=1.0, lambda_2=1.0,
            k2=5, k3=5, k3_bar=5, dims=500):
    """
    Peform end-to-end processing.
    """
    logger.info("Source = %s, Target = %s" % (source_domain, target_domain))
    base_path = "../work/%s-%s" % (source_domain, target_domain)
    Ua = load_matrix("%s/Ua.mtx" % base_path)
    Ub = load_matrix("%s/Ub.mtx" % base_path)
    A = load_matrix("%s/A.mtx" % base_path)
    B = load_matrix("%s/B.mtx" % base_path)
    XlA_pos = load_matrix("%s/XlA_pos.mtx" % base_path)
    XlA_neg = load_matrix("%s/XlA_neg.mtx" % base_path)
    XuA = load_matrix("%s/XuA.mtx" % base_path)
    XuB = load_matrix("%s/XuB.mtx" % base_path)

    logger.info("w1 = %f, w2 = %f, w3 = %f" % (w1, w2, w3))
    logger.info("dims = %d" % dims)
    logger.info("l1 = %f, k2 = %d" % (lambda_1, k2))
    logger.info("l2 = %f, k3 = %d, k3' = %d" % (lambda_2, k3, k3_bar))

    SE = SupEmb(Ua, Ub, A, B, XlA_pos, XlA_neg, XuA, XuB)
    SE.w1 = w1 
    SE.w2 = w2 
    SE.w3 = w3
    SE.lambda_1 = lambda_1 
    SE.lambda_2 = lambda_2 
    SE.k2 = k2 
    SE.k3 = k3 
    SE.k3_bar = k3_bar 
    SE.dims = dims
    Q = SE.get_embedding()
    #SE.save_embedding("%s/Q.mtx" % base_path, Q)
    #Q = SE.load_embedding("%s/Q.mtx" % base_path)
    #SE.check_symmetry(Q)

    # check whether there are zero columns or rows in Q.
    Pa, Pb = SE.get_projection(Q)
    pos_train = SE.project_instances(XlA_pos, Ua, A, Pa)
    neg_train = SE.project_instances(XlA_neg, Ua, A, Pa)
    pivots = SE.load_pivots("../work/%s-%s/DI_list" % (source_domain, target_domain))
    source_specific_feats = SE.load_feature_space("../work/%s-%s/source_specific_feats" % (source_domain, target_domain))
    target_specific_feats = SE.load_feature_space("../work/%s-%s/target_specific_feats" % (source_domain, target_domain))
    source_feats = pivots[:]
    source_feats.extend(source_specific_feats)
    target_feats = pivots[:]
    target_feats.extend(target_specific_feats)
    feat_index = pivots[:]
    for w in source_specific_feats:
        if w not in feat_index:
            feat_index.append(w)
    for w in target_specific_feats:
        if w not in feat_index:
            feat_index.append(w)
    pos_train = SE.concatenate_original_projected(pos_train, XlA_pos, source_feats, feat_index)
    neg_train = SE.concatenate_original_projected(neg_train, XlA_neg, source_feats, feat_index)
    model = train_logistic(pos_train, neg_train)
    XlB_pos = load_matrix("%s/XlB_pos.mtx" % base_path)
    XlB_neg = load_matrix("%s/XlB_neg.mtx" % base_path)
    pos_test = SE.project_instances(XlB_pos, Ub, B, Pb)
    neg_test = SE.project_instances(XlB_neg, Ub, B, Pb)
    pos_test = SE.concatenate_original_projected(pos_test, XlB_pos, target_feats, feat_index)
    neg_test = SE.concatenate_original_projected(neg_test, XlB_neg, target_feats, feat_index)
    return test_logistic(pos_test, neg_test, model)


def no_adapt_baseline(source_domain, target_domain):
    """
    Implements the no adapt baseline. Train a classifier using source domain
    labeled instances and then evaluate it on target domain test instances.
    """
    base_path = "../work/%s-%s" % (source_domain, target_domain)
    XlA_pos = load_matrix("%s/XlA_pos.mtx" % base_path)
    XlA_neg = load_matrix("%s/XlA_neg.mtx" % base_path)
    XlB_pos = load_matrix("%s/XlB_pos.mtx" % base_path)
    XlB_neg = load_matrix("%s/XlB_neg.mtx" % base_path)
    SE = SupEmb()
    pivots = SE.load_pivots("../work/%s-%s/DI_list" % (source_domain, target_domain))
    source_specific_feats = SE.load_feature_space("../work/%s-%s/source_specific_feats" % (source_domain, target_domain))
    target_specific_feats = SE.load_feature_space("../work/%s-%s/target_specific_feats" % (source_domain, target_domain))
    source_feats = pivots[:]
    source_feats.extend(source_specific_feats)
    target_feats = pivots[:]
    target_feats.extend(target_specific_feats)
    feat_index = pivots[:]
    for w in source_specific_feats:
        if w not in feat_index:
            feat_index.append(w)
    for w in target_specific_feats:
        if w not in feat_index:
            feat_index.append(w)
    pos_train = SE.concatenate_original_projected(None, XlA_pos, source_feats, feat_index)
    neg_train = SE.concatenate_original_projected(None, XlA_neg, source_feats, feat_index)
    model = train_logistic(pos_train, neg_train)
    pos_test = SE.concatenate_original_projected(None, XlB_pos, target_feats, feat_index)
    neg_test = SE.concatenate_original_projected(None, XlB_neg, target_feats, feat_index)
    test_logistic(pos_test, neg_test, model)
    pass


def batch_mode(source, target, stat_file):
    """
    Runs SupEmb using different parameter settings. 
    """
    global debug_file
    lambda_1_vals = [0, 1.0, 10000.0]
    lambda_2_vals = [0, 1.0, 10000.0]
    k2_vals = [1, 5, 9] # k for k-NN when computing W2
    k3_vals = [1, 5, 9] # k for k-NN when computing W3
    k3_bar_vals = [1, 5, 9] # k for k-NN when computing W3_bar
    dims_vals = [10, 20, 30] # number of latent dimensions for the embedding
    res_file = open("../work/batch_%s_%s.csv" % (source, target), 'w')
    res_file.write("w1, w2, w3, l1, l2, k2, k3, k3', dims, acc\n")
    res_file.flush()

    debug_file = open("../work/debug/%s_%s_debug.csv" % (source, target), "w")
    debug_file.write("# O1, O2, O3\n")

    # Running 3 settings for Rule 1.
    best_rule1 = {}
    best_rule1_acc = 0
    for dims in dims_vals:
        w2 = w3 = 0
        l1 = l2 = k2 = k3 = k3_bar = 1
        w1 = 1
        acc = process(source, target, w1, w2, w3, l1, l2, k2, k3, k3_bar, dims)
        res_file.write("%f, %f, %f, %f, %f, %d, %d, %d, %d, %f\n" % (w1, w2, w3, l1, l2, k2, k3, k3_bar, dims, acc))
        res_file.flush()
        if acc > best_rule1_acc:
            best_rule1_acc = acc 
            best_rule1 = {'w1':w1, 'w2':w2, 'w3':w3, 'l1':l1, 'l2':l2, 'k2':k2, 'k3':k3, 'k3_bar':k3_bar, 'dims':dims}
    stat_file.write("%s, %s, %f, %d, " % (source, target, best_rule1_acc, best_rule1["dims"]))
    stat_file.flush()

    # Running 27 settings for Rule 2.
    best_rule2 = {}
    best_rule2_acc = 0
    for dims in dims_vals:
        for l1 in lambda_1_vals:
            for k2 in k2_vals:
                w1 = w3 = 0
                l2 = k3 = k3_bar = 1
                w2 = 1
                acc = process(source, target, w1, w2, w3, l1, l2, k2, k3, k3_bar, dims)
                res_file.write("%f, %f, %f, %f, %f, %d, %d, %d, %d, %f\n" % (w1, w2, w3, l1, l2, k2, k3, k3_bar, dims, acc))
                res_file.flush()
                if acc > best_rule2_acc:
                    best_rule2_acc = acc 
                    best_rule2 = {'w1':w1, 'w2':w2, 'w3':w3, 'l1':l1, 'l2':l2, 'k2':k2, 'k3':k3, 'k3_bar':k3_bar, 'dims':dims}
    stat_file.write("%f, %f, %d, %d, " % (best_rule2_acc, best_rule2["l1"], best_rule2["k2"], best_rule2["dims"]))
    stat_file.flush()

    # Running 81 settings for Rule 3.
    best_rule3 = {}
    best_rule3_acc = 0
    for dims in dims_vals:
        for l2 in lambda_2_vals:
            for k3 in k3_vals:
                for k3_bar in k3_bar_vals:
                    w1 = w2 = 0
                    l1 = k2 = 1
                    w3 = 1
                    acc = process(source, target, w1, w2, w3, l1, l2, k2, k3, k3_bar, dims)
                    res_file.write("%f, %f, %f, %f, %f, %d, %d, %d, %d, %f\n" % (w1, w2, w3, l1, l2, k2, k3, k3_bar, dims, acc))
                    res_file.flush()
                    if acc > best_rule3_acc:
                        best_rule3_acc = acc 
                        best_rule3 = {'w1':w1, 'w2':w2, 'w3':w3, 'l1':l1, 'l2':l2, 'k2':k2, 'k3':k3, 'k3_bar':k3_bar, 'dims':dims}
    stat_file.write("%f, %f, %d, %d, %d\n" % (best_rule3_acc, best_rule3["l2"], best_rule3["k3"], best_rule3["k3_bar"], best_rule3["dims"]))
    stat_file.flush()

    # Run 12 settings with the best parameter values for the three rules.
    for (w1, w2, w3) in [(1,1,0), (1,0,1), (0,1,1), (1,1,1)]:
        for dims in dims_vals:
            k2 = best_rule2["k2"]
            l1 = best_rule2["l1"]
            l2 = best_rule3["l2"]
            k3 = best_rule3["k3"]
            k3_bar = best_rule3["k3_bar"]
            acc = process(source, target, w1, w2, w3, l1, l2, k2, k3, k3_bar, dims)
            res_file.write("%f, %f, %f, %f, %f, %d, %d, %d, %d, %f\n" % (w1, w2, w3, l1, l2, k2, k3, k3_bar, dims, acc))
            res_file.flush()
    res_file.close()    
    debug_file.close()
    pass


def run_batch_mode(source, target):
    stat_file = open("../work/debug/%s_%s_stats.csv" % (source, target), 'w')
    stat_file.write("#source, target, R1, d, R2, l1, k2, d, R3, l2, k3, k3', d\n")
    stat_file.flush() 
    batch_mode(source, target, stat_file)
    stat_file.close()   
    pass


def batch_source_fixed():
    source = sys.argv[1].strip()
    domains = ["dvd", "books", "electronics", "kitchen"]
    for target in domains:
        if source != target:
            run_batch_mode(source, target)
    pass

if __name__ == "__main__":
    #batch_source_fixed()
    run_batch_mode("dvd", "kitchen")
    #source_domain = "testSource"
    #target_domain = "testTarget"
    #no_adapt_baseline(source_domain, target_domain)     
    #process(source_domain, target_domain)
    


