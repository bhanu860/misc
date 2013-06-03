'''
Created on May 27, 2013

@author: bhanu
'''
import unittest
import guess_language as gl
import langid
from lid import Lid

class Test(unittest.TestCase):
    def test_guess_langauge(self):
        with open("input.txt", 'r') as textfile:
            print "guess_language: "
            for text in textfile:
                print gl.guessLanguageName(text)
            print
    
    
    def test_langid(self):
        with open("input.txt", 'r') as textfile:
            print "langid: "
            for text in textfile:
                print langid.classify(text)
            print
            
#    def test_lid(self):
#        with open("input.txt", 'r') as textfile:
#            print "lid: "
#            lid = Lid()
#            for text in textfile:
#                print lid.checkText(text)
#            print


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_guess_langauge']
    unittest.main()