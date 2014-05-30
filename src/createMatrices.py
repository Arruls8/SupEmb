"""
Load the data for a pair of domains and create the input matrices 
required by the supervised embedding (SupEmb) method. 
"""

__author__ = 'Danushka Bollegala'
__license__ = 'BSD'
__date__ = "16/04/2014"

import scipy.sparse
from scipy.io import mmwrite


class DATA:

    def __init__(self, source_domain, target_domain, d, h):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.d = d
        self.h = h
        self.IGNORE_FEATS_TH = 0
        self.base_dir = "../SFA/work"
        pass


    def load_data(self):
        """
        Loads the data for the specified domain pair. 
        """
        self.load_pivots("%s/%s-%s/DI_list" % (self.base_dir, self.source_domain, self.target_domain))
        self.load_nonpivots("%s/%s-%s/DS_list.1000" % (self.base_dir, self.source_domain, self.target_domain))

        # Load source and target, train and test documents.
        self.src_train_pos_docs =self.load_documents("%s/%s/train.positive" % (self.base_dir, self.source_domain))
        self.src_train_neg_docs =self.load_documents("%s/%s/train.negative" % (self.base_dir, self.source_domain))
        self.src_train_unlab_docs =self.load_documents("%s/%s/train.unlabeled" % (self.base_dir, self.source_domain))
        self.tgt_train_unlab_docs = self.load_documents("%s/%s/train.unlabeled" % (self.base_dir, self.target_domain))
        self.tgt_test_pos_docs =self.load_documents("%s/%s/test.positive" % (self.base_dir, self.target_domain))
        self.tgt_test_neg_docs =self.load_documents("%s/%s/test.negative" % (self.base_dir, self.target_domain))
        self.NlA_pos = len(self.src_train_pos_docs)
        self.NlA_neg = len(self.src_train_neg_docs)
        self.NuA = len(self.src_train_unlab_docs)
        self.NuB = len(self.tgt_train_unlab_docs)
        self.NlA = self.NlA_pos + self.NlA_neg
        
        # Create the complete train documents for the source
        self.src_train_docs = self.src_train_pos_docs[:]
        self.src_train_docs.extend(self.src_train_neg_docs)
        self.src_train_docs.extend(self.src_train_unlab_docs)

        "Computing source domain feature space..."
        self.src_feats = self.get_domain_feats(self.src_train_docs, self.pivots, self.source_specific_feats, self.d)
        print "Computing target domain feature space..."
        self.tgt_feats = self.get_domain_feats(self.tgt_train_unlab_docs, self.pivots, self.target_specific_feats, self.h)


        # Create document representations.
        self.XlA_pos = self.get_doc_vect(self.src_train_pos_docs, self.pivots, self.source_specific_feats, self.src_feats)
        self.XlA_neg = self.get_doc_vect(self.src_train_neg_docs, self.pivots, self.source_specific_feats, self.src_feats)
        self.XuA = self.get_doc_vect(self.src_train_unlab_docs, self.pivots, self.source_specific_feats, self.src_feats)
        self.XuB = self.get_doc_vect(self.tgt_train_unlab_docs, self.pivots, self.target_specific_feats, self.tgt_feats)
        self.XlB_pos = self.get_doc_vect(self.tgt_test_pos_docs, self.pivots, self.target_specific_feats, self.tgt_feats)
        self.XlB_neg = self.get_doc_vect(self.tgt_test_neg_docs, self.pivots, self.target_specific_feats, self.tgt_feats)

        # Create the complete train documents for the source
        self.src_train_docs = self.src_train_pos_docs[:]
        self.src_train_docs.extend(self.src_train_neg_docs)
        self.src_train_docs.extend(self.src_train_unlab_docs)

        self.Ua, self.A = self.get_feat_representations(self.pivots, self.source_specific_feats, self.src_feats, self.src_train_docs)
        self.Ub, self.B = self.get_feat_representations(self.pivots, self.target_specific_feats, self.tgt_feats, self.tgt_train_unlab_docs)
        pass


    def save_matrices(self, base_path):
        """
        Save the matrices to disk.
        """
        mmwrite("%s/Ua.mtx" % base_path, self.Ua)
        mmwrite("%s/A.mtx" % base_path, self.A)
        mmwrite("%s/Ub.mtx" % base_path, self.Ub)
        mmwrite("%s/B.mtx" % base_path, self.B)
        mmwrite("%s/XlA_pos.mtx" % base_path, self.XlA_pos)
        mmwrite("%s/XlA_neg.mtx" % base_path, self.XlA_neg)
        mmwrite("%s/XuA.mtx" % base_path, self.XuA)
        mmwrite("%s/XuB.mtx" % base_path, self.XuB)
        mmwrite("%s/XlB_pos.mtx" % base_path, self.XlB_pos)
        mmwrite("%s/XlB_neg.mtx" % base_path, self.XlB_neg)
        pass


    def get_doc_vect(self, docs, pivots, nonpivots, feat_space):
        """
        Get a vectors representing the documents using the union of pivots and 
        nonpivots as the feature space.
        Also remove from the docs any features that do not appear in the union of 
        pivous, nonpivots, and feat_space.
        """
        S = pivots[:]
        S.extend(nonpivots)
        X = scipy.sparse.lil_matrix((len(docs), len(S)), dtype=float)
        for (i, doc) in enumerate(docs):
            for word in doc:
                if word in S:
                    j = S.index(word)
                    X[i,j] = 1
        return X


    def get_domain_feats(self, train_docs, pivots, specific_feats, n):
        """
        For each feature that appear in train documents, compute its co-occurrences with 
        pivots or domain specific features for the domain under consideration. 
        Sort the features in the descending order of their co-occurrences computed above
        and select the top ranked n features for representing the domain. 
        """

        candidates = set(pivots).union(set(specific_feats))
        total_freq = {}
        for doc in train_docs:
            for feat in doc:
                total_freq[feat] = total_freq.get(feat, 0) + 1
        counts = {}
        for doc in train_docs:
            for feat in doc:
                if total_freq[feat] < self.IGNORE_FEATS_TH:
                    continue
                rest = doc - set(feat)
                counts[feat] = counts.get(feat, 0) + len(rest.intersection(candidates))
        counts_list = counts.items()
        counts_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        print "Selecting %d out of %d" % (n, len(counts_list))
        print "Top 5 are:", counts_list[:5]
        return [x[0] for x in counts_list[:n]]



    def load_documents(self, fname):
        """
        Load reviews. 
        """
        F = open(fname)
        L = []
        for line in F:
            L.append(set(line.strip().split()))
        F.close()
        return L


    def load_pivots(self, pivots_filename):
        self.pivots = []
        pivots_file = open(pivots_filename)
        for line in pivots_file:
            self.pivots.append(line.strip().split()[1])
        pivots_file.close()
        self.M = len(self.pivots)
        pass


    def load_nonpivots(self, DS_filename):
        self.source_specific_feats = []
        self.target_specific_feats = []
        DS_file = open(DS_filename)
        for line in DS_file:
            p = line.strip().split()
            if p[2] == 'S':
                self.source_specific_feats.append(p[1].strip())
            elif p[2] == 'T':
                self.target_specific_feats.append(p[1].strip())
            else:
                print "Invalid domain label. Must be S or T"
                raise ValueError
        DS_file.close()
        self.MA = len(self.source_specific_feats)
        self.MB = len(self.target_specific_feats)
        pass


    def get_feat_representations(self, pivots, nonpivots, feat_space, docs):
        """
        Create a matrix between pivots and domain features. 
        Also creates a matrix between nonpivots and domain features. 
        """
        n_pivots = len(pivots)
        n_nonpivots = len(nonpivots)
        n_feats = len(feat_space)
        U = scipy.sparse.lil_matrix((n_pivots, n_feats), dtype=float)
        A = scipy.sparse.lil_matrix((n_nonpivots, n_feats), dtype=float)
        for (i, doc) in enumerate(docs):
            for word in doc:
                if word in pivots:
                    for w in doc:
                        if w in feat_space and word != w:
                            U[pivots.index(word), feat_space.index(w)] += 1
                if word in nonpivots:
                    for w in doc:
                        if w in feat_space and word != w:
                            A[nonpivots.index(word), feat_space.index(w)] += 1
        return U, A

    pass



def process(source, target):
    D = DATA(source, target, 1000, 900)
    D.load_data()
    print "No. of pivots =", D.M
    print "Source domain specific feats Ma =", D.MA
    print "Target domain specific feats Mb =", D.MB
    print "Source Train Docs: pos = %d, neg = %d unlabeled = %d" % (D.NlA_pos, D.NlA_neg, D.NuA)
    print "Target Train Docs: unlabeled = %d" % (D.NuB)
    base_path = "../work/%s-%s" % (D.source_domain, D.target_domain)
    D.save_matrices(base_path)
    pass

if __name__ == "__main__":
    process("books", "electronics")
    #process("testSource", "testTarget")





