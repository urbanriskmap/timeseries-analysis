import nlp.aws_nlp as nlp
import nlp.util as util
import pickle

def get_and_pickle_all_sents(filename='./sents.p'):
    text = nlp.get_all_report_text()
    print(text.index.names)
    print(next(text.iterrows()))
    def save(sents):
        pickle.dump(sents, open(filename, 'wb'))

    sents = nlp.get_all_sentiments(text, hook=save)
    for index, row in text.iterrows():
        print(row['text'],  nlp.get_sentiment(row['text']))

def load_pickled_sents(filename='./nlp/sents.p'):
    return pickle.load(open(filename, "rb"))

def p_flood_given_neg(sents, flood_keys, non_flood_keys):
    '''
    probabilty of flooding given a negative sentiment
    estimated as:
    = # flood and neg / (# negative)
    '''

    print(flood_keys)
    no_flood_and_neg = 0
    flood_and_neg = 0

    no_flood_and_pos = 0 
    flood_and_pos = 0

    neg = 0
    pos = 0
    for key,value in sents.items():
        if value['SentimentScore']['Negative'] > .50:
            neg += 1
            # print('negative pkey: ', key)
            # print(value)
            if key in flood_keys.index:
                flood_and_neg += 1
            else: 
                no_flood_and_neg += 1
        else:
            pos += 1
            if key in flood_keys.index:
                flood_and_pos += 1
            else: 
                no_flood_and_pos += 1



    # true positive rate 
    total_num_of_reports = len(sents)
    print("True positive")
    print(flood_and_neg/neg)

    # false positive: negative sentiment, but no flood
    # P ( no flood | neg sentiment)  = 
    # noflood and neg sent / (neg)
    print("False positive")
    print(no_flood_and_neg/neg)

    return 

if __name__ == '__main__':
    # get_and_pickle_all_sents()
    sents = load_pickled_sents()

    start_known_flood = "'2017-11-01 00:00:35.630000-04:00'"
    end_known_flood = "'2017-11-07 00:00:35.630000-04:00'"

    flood_keys = util.get_flood_pkeys(start_known_flood, end_known_flood, nlp.engine)
    non_flood_keys = util.get_no_flood_pkeys(start_known_flood, end_known_flood, nlp.engine)

    p_flood_given_neg(sents, flood_keys, non_flood_keys)
