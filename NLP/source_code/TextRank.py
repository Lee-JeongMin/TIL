# 'summarizer' module provides functions for summarizing texts. 
# Summarizing is based on ranks of text sentences using a variation 
# of the TextRank algorithm.
#
# Federico Barrios, et, al., 2016, Variations of the Similarity Function 
# of TextRank for Automated Summarization, https://arxiv.org/abs/1602.03606
#
# Barrios는 tfidf 대신 BM25, BM25+를 사용했고, cosine similarity를 사용했다.
# gensim.summarizer도 Barrios의 TextRank를 사용한다.
from gensim.summarization.summarizer import summarize

text = \
'''Rice Pudding - Poem by Alan Alexander Milne
What is the matter with Mary Jane?
She's crying with all her might and main,
And she won't eat her dinner - rice pudding again -
What is the matter with Mary Jane?
What is the matter with Mary Jane?
I've promised her dolls and a daisy-chain,
And a book about animals - all in vain -
What is the matter with Mary Jane?
What is the matter with Mary Jane?
She's perfectly well, and she hasn't a pain;
But, look at her, now she's beginning again! -
What is the matter with Mary Jane?
What is the matter with Mary Jane?
I've promised her sweets and a ride in the train,
And I've begged her to stop for a bit and explain -
What is the matter with Mary Jane?
What is the matter with Mary Jane?
She's perfectly well and she hasn't a pain,
And it's lovely rice pudding for dinner again!
What is the matter with Mary Jane?
'''
# ratio (float, optional) – Number between 0 and 1 that determines the 
# proportion of the number of sentences of the original text to be chosen 
# for the summary.
s = summarize(text, ratio = 0.2)
print(summarize(text))



