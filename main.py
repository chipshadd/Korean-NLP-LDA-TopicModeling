import pandas as pd
from konlpy.tag import Okt; t = Okt()
# from konlpy.tag import Okt
import nltk
from konlpy.corpus import kobill
from gensim import corpora, models, parsing
from gensim import models
import numpy as np; np.random.seed(42)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import argparse
import pickle
import yaml
from collections import Counter

def clean(arr):
    texts_ko = []
    total_length = 0
    for item in arr:
        new = []
        for word in item:
            split = word.split("/")
            if split[1] == 'Noun':
                if len(split[0]) > 1:
                    if split[0] not in stop_words:
                        new.append(word)
        texts_ko.append(new)
    return texts_ko

def train(data=None,init=True): # Always true if you want ephemeral modeling
    dfs = []
    if data is None:
        dir_path = './training_data'
        files = os.listdir(dir_path)
        excel_files = [file for file in files if file.endswith('.xlsx') or file.endswith('.xls')]
        for file in excel_files:
            file_path = os.path.join(dir_path, file)
            dfs.append(pd.read_excel(file_path))
    else:
        dfs.append(data)
    
    # Clean up text
    docs_ko = []
    for df in dfs:
        for x in df[column_name]:
            docs_ko.append(parsing.strip_punctuation(x))
        # docs_ko.append([x.replace('.', '').replace(',','').replace("'","").replace('·', ' ').replace('=','').replace('\n','') for x in df[column_name]])
    pos = lambda d: ['/'.join(p) for p in t.pos(d, stem=True, norm=True)]
    tmp = [pos(doc) for doc in docs_ko]
    texts_ko = clean(tmp)

    # Tokenize words to IDs and merge with existing dictionary
    if init:
        dict_ko = corpora.Dictionary(texts_ko)
    else:
        dict_ko = corpora.Dictionary.load('ko.dict')
        # dict_ko = corpora.Dictionary.load('ko_lda.id2word')

    # Get new text frequency (TFIDF)
    tf_ko = [dict_ko.doc2bow(text, allow_update=True) for text in texts_ko]
    tfidf_model_ko = models.TfidfModel(tf_ko)
    tfidf_ko = tfidf_model_ko[tf_ko]
    # Load and merge Store in Matrix Market format
    corpora.MmCorpus.serialize('ko.mm', tfidf_ko) 
    dict_ko.save('ko.dict')
    if init:
        # lda_ko = models.ldamodel.LdaModel(tfidf_ko, num_topics=ntopics, passes=npasses, chunksize=chunk_size, update_every=update_frequency)
        lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dict_ko, num_topics=ntopics, passes=npasses, chunksize=chunk_size, update_every=update_frequency)
        lda_ko.save('ko_lda.lda')
    else:
        lda_ko = models.ldamodel.LdaModel(id2word=dict_ko, num_topics=ntopics)
        lda_ko.load('ko_lda.lda')
        # lda = models.LdaModel.load('ko_lda')
        # lda.id2word = dict_ko
        lda_ko.sync_state()
        lda_ko.update(tfidf_ko, passes=npasses, chunksize=chunk_size, update_every=update_frequency)
        lda_ko.save('ko_lda.lda')

    # get_info(lda_ko)
    print("Finished building the model")

def get_info(lda=None):
    if lda is None:
        lda = models.LdaModel.load('ko_lda.lda')
    print("Topics modeled:")
    for topic in lda.print_topics(num_topics=ntopics, num_words=nwords):
        print(topic)

def analyze(file): # Pass in df
    print("Generating topic models....")
    train(data=file)
    lda_ko = models.ldamodel.LdaModel.load('ko_lda.lda')
    pos = lambda d: ['/'.join(p) for p in t.pos(d, stem=True, norm=True)]
    docs_ko = [parsing.strip_punctuation(x) for x in file[column_name]]
    tmp = [pos(doc) for doc in docs_ko]
    texts_ko = clean(tmp)
    dict_ko = corpora.Dictionary.load('ko.dict')
    # dict_ko = corpora.Dictionary.load('ko_lda.id2word')
    # dict_ko.merge_with(corpora.Dictionary(texts_ko))

    # print(dict_ko.get(2682))
    # print(dict_ko.doc2bow(texts_ko[213]))
    # print(212, sorted(lda_ko.get_document_topics(dict_ko.doc2bow(texts_ko[213])), key=lambda x: x[1], reverse=True))

    print()
    c = Counter()
    flattened_words = []
    for words in texts_ko:
        for word in words:
            c[word] += 1
            flattened_words.append(word.split("/")[0])
    print("Generating data....")
    for k,v in c.most_common(30):
        print(f'{k.split("/")[0]},{v}')

    # print(c.most_common(30))

    c_topic = Counter()
    for i in range(len(texts_ko)):
        found_topic = sorted(lda_ko.get_document_topics(dict_ko.doc2bow(texts_ko[i], allow_update=True)), key=lambda x: x[1], reverse=True)[0][0]
        c_topic[found_topic] +=1
        # print(i, sorted(lda_ko.get_document_topics(dict_ko.doc2bow(texts_ko[i], allow_update=True)), key=lambda x: x[1], reverse=True)[0][0])

    print()
    print(f'{len(texts_ko)} articles found.')
    for topic,count in c_topic.most_common(ntopics):
        print(f'{count} were about topic #{topic}')

    print()
    get_info(lda_ko)

    # wordcloud
    FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
    dictionary_ko = corpora.Dictionary.load('ko.dict')

    all_words = ' '.join(flattened_words)
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=FONT_PATH).generate(all_words)

    # Plot the WordCloud image                        
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


    # for article in texts_ko:
    #     bow = tfidf_model_ko[dictionary_ko.doc2bow(article)]
    #     print(sorted(lda_ko[bow], key=lambda x: x[1], reverse=True))

def load_parameters(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        parameters = yaml.safe_load(yaml_file)
    return parameters

if __name__ == "__main__":
    yaml_file_path = "config.yml"
    parameters = load_parameters(yaml_file_path)
    column_name = u'본문'  # Expecting excel sheets from BigKinds, this column should contain the body of the article
    ntopics = parameters.get("lda", {}).get("number_of_topics")
    nwords = parameters.get("lda", {}).get("number_of_words_per_topic")
    npasses = parameters.get("lda", {}).get("number_of_training_passes")
    chunk_size = parameters.get("lda", {}).get("chunk_size")
    stop_words = parameters.get("main", {}).get("stop_words")
    update_frequency = parameters.get("lda", {}).get("update_frequency")
    training_directory = parameters.get("main", {}).get("training_directory")
    parser = argparse.ArgumentParser(description="Parse arguments for LDA functions.")
    # parser.add_argument("sub", choices=['analyze', 'train', 'info'])
    parser.add_argument("sub", choices=['analyze', 'info'])
    parser.add_argument("--file", default=None)
    args = parser.parse_args()
    if args.sub == 'analyze' and args.file is not None:
        df = pd.read_excel(args.file)
        analyze(df)
    # if args.sub == 'train':
    #     train()
        # train(init=True)
    if args.sub == 'info':
        get_info()
    # parse()