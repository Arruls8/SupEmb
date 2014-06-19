"""
This scrip implements the unit tests for SupEmb. 
"""

__author__ = "Danushka Bollegala"
__date__ = "18/04/2014"


from nose.tools import *
import numpy
from numpy.testing import assert_array_equal


class TestDATA():

    def __init__(self):
        from createMatrices import DATA
        self.D = DATA("testSource", "testTarget", 7, 3)
        self.D.load_data()
        print "No. of pivots = %d "% self.D.M
        print "Source domain specific feats Ma = %d" % self.D.MA
        print "Target domain specific feats Mb = %d" % self.D.MB
        print "Source Train Docs: pos = %d, neg = %d unlabeled = %d" % (self.D.NlA_pos, self.D.NlA_neg, self.D.NuA)
        print "Target Train Docs: unlabeled = %d" % self.D.NuB
        base_path = "../work/%s-%s" % (self.D.source_domain, self.D.target_domain)
        self.D.save_matrices(base_path)
        pass

    def test_pivots(self):
        """
        Testing for pivots
        """
        assert_equal(self.D.pivots, ["likes"])

    def test_source_nonpivots(self):
        """
        Testing for source nopivots
        """
        assert_equal(self.D.source_specific_feats, ["john", "apples", "david", "hates", "are", "red"])

    def test_target_nonpivots(self):
        """
        Testing for target nonpivots
        """
        assert_equal(self.D.target_specific_feats, ["simon", "bananas"])

    def test_source_feat_space(self): 
        """
        Testing for source feature space
        """
        assert_equal(self.D.src_feats, ["apples", "are", "david", "likes", "john", "hates", "red"])

    def test_target_feat_space(self):
        """
        Testing for target feature space
        """
        assert_equal(self.D.tgt_feats, ['simon', 'likes', 'bananas'])

    def test_Ua(self):
        """
        Testing Ua
        """
        print self.D.pivots 
        print self.D.src_feats
        print self.D.Ua.todense() 
        Ua = numpy.array([[ 1,  0,  0, 0, 1, 0, 0]])
        assert_array_equal(self.D.Ua.toarray(), Ua)

    def test_A(self):
        """
        Testing A
        """
        print self.D.source_specific_feats
        print self.D.src_feats
        print self.D.A.todense() 
        A = numpy.array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.],
                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.],   
                         [ 1.,  0.,  0.,  0.,  0.,  1.,  0.],
                         [ 1.,  0.,  1.,  0.,  0.,  0.,  0.],
                         [ 1.,  0.,  0.,  0.,  0.,  0.,  1.],
                         [ 1.,  1.,  0.,  0.,  0.,  0.,  0.]])
        assert_array_equal(self.D.A.toarray(), A)

    def test_Ub(self):
        """
        Testing Ub
        """
        print self.D.pivots
        print self.D.tgt_feats
        print self.D.Ub.todense() 
        Ub = numpy.array([[1, 0, 1]])
        assert_array_equal(self.D.Ub.toarray(), Ub)

    def test_B(self):
        """
        Testing B
        """
        print self.D.target_specific_feats
        print self.D.tgt_feats
        print self.D.B.todense() 
        B = numpy.array([[0., 1., 1.], [1., 1., 0.]])
        assert_array_equal(self.D.B.toarray(), B)

    def test_XlA_pos(self):
        """
        Testing XlA_pos
        """
        print self.D.pivots
        print self.D.src_train_pos_docs
        print self.D.src_feats
        print self.D.XlA_pos.toarray()
        XlA_pos = numpy.array([[1, 0, 0, 1, 1, 0, 0]])
        assert_array_equal(self.D.XlA_pos.toarray(), XlA_pos)

    def test_XlA_neg(self):
        """
        Testing XlA_neg
        """
        print self.D.src_train_neg_docs
        print self.D.src_feats
        print self.D.XlA_neg.toarray()
        XlA_neg = numpy.array([[1, 0, 1, 0, 0, 1, 0]])
        assert_array_equal(self.D.XlA_neg.toarray(), XlA_neg)

    def test_XuA(self):
        """
        Testing XuA
        """
        print self.D.src_train_unlab_docs
        print self.D.src_feats
        print self.D.XuA.toarray()
        XuA = numpy.array([[1 , 1, 0, 0, 0, 0, 1]])
        assert_array_equal(self.D.XuA.toarray(), XuA)

    def test_XuB(self):
        """
        Testing XuB
        """
        print self.D.src_train_unlab_docs
        print self.D.tgt_feats
        print self.D.XuB.toarray()
        XuB = numpy.array([[1, 1, 1]])
        assert_array_equal(self.D.XuB.toarray(), XuB)       

    pass

