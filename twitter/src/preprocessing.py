'''
Created on May 27, 2013

@author: bhanu
'''


import tweet_parser as parser
import langid
from ttp import ttp



def extract_tweet(filename, min_num_chars=10):
    with open(filename, 'r') as tweetfile:
        for tweet in tweetfile:            
            tweettext = parser.get_tweet_text(tweet,remove_urls=True, remove_retweets=True, remove_usernames=True)
            if(len(tweettext) > min_num_chars):
                tweetlang = langid.classify(tweettext)
                print tweettext[0:50] if len(tweettext) > 50 else  tweettext , " :    ", tweetlang , '\n'
            
        


def extract_tweet_tags(filename):
    with open(filename, 'r') as tweetfile:
        for tweet in tweetfile:     
            result = ttp.Parser().parse(tweet.strip(), html=False)      
            print result.users, "    ", result.tags, "    ", result.urls, "\n" 
        

if __name__ == "__main__":    
    #extract_tweet("twitter12051154249.txt")
    extract_tweet_tags("twitter12051154249.txt")
    