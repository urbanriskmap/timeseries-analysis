import pandas as pd
from sqlalchemy.sql import text

import numpy as np


class CognicityTextLoader:

    def __init__(self, configObj):
        """ Creates a data loader for cognicity textual report data
        Args:
            configObject(dict)
                databaseEngine: a sqlalchemy database engine
                location: a location string, one of "id" or "ch"
                img_folder_prefix: path for folder to store downloaded images
                logger: a python logging object
        Returns:
            None
        """
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.img_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]
        self.logger.debug("CognicityTextLoader constructed")

    def get_data(self):
        """
        Returns:
            pandas dataframe of pkey to all the reports in the database
        """
        rows = pd.read_sql_query(text('''
        SELECT pkey, text from ''' + self.database_name + '''.all_reports
            WHERE text IS NOT null
            AND pkey > 119
            AND LENGTH(text) > 0
            AND text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%'
            ORDER BY created_at
        '''), con=self.database, index_col="pkey")
        return rows

    def prepare_text(inp):
        '''
        returns a list of strings where each string is a different token
        '''
        # TODO replace this with regex?
        return inp.lower().replace('.', ' ').replace(',', ' ').split()
        
        
    def count_frequency(pkey_text_df):
        '''
        Params: 
            pkey_text_df: pandas dataframe with index as pkey and one column named 'text'
        
        Returns:
            a dictionary of (str:int) that represents how often each 
            string is repeated in the text
        '''
        # TODO do we really need this?
        pass
    
    def make_unary_vocab(pkey_text_df):
        # go through all report texts creating set
        # of all possible words
        vocab = dict()
        occur = dict()
        index = 0
        reports_dict = dict()
        for row in pkey_text_df.iterrows():
            report_text_list = prepare_text(row[1]['text'])
            pkey = row[0]
            reports_dict[pkey] = report_text_list
            for word in report_text_list:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                if word not in occur:
                    occur[word] = 1
                else:
                    occur[word] += 1
        return (vocab, occur, reports_dict)
                
    def make_unary_feature_vector(vocab, report_text_list):
        res = np.zeros((len(vocab), 1))
        for word in report_text_list:
            if word in vocab:
                res[vocab[word]][0] = 1
        return res
    
    def make_bigram_feature_vector(vocab, report_text_list):
        # needs a tuple vocab[(tup)] -> index
        res = np.zeros((len(vocab), 1))
        for first, second in zip(report_text_list, report_text_list[1:]):
            res[vocab[(first, second)]] = 1
        
        return res
            
    
    def make_feature_matrix(vocab, reports_dict, pos_pkeys, neg_pkeys):
        """
        params:
            vocab (dict): dictionary from word to index location along the
                      column vector
            report_texts: dictionary from pkeys to text
            pos_pkeys (set): membership in this set means the pkey should be 
                             labeled as postive
        Returns:
            tuple of (feature_matrix, labels)
            feature_matrix is a matrix of pkeys with associated feature columns underneath
            pkey      | pkey
            feat_vect | feat
            size: (len(vocab)+1, num_reports)
            labels is a matrix of size (1, num_reports) where report i is +1 if 
                pkey matching that index is in pos_pkeys and -1 else
        """
        
        labels = np.zeros((1, (len(reports_dict))))
        feature_matrix = np.zeros((len(vocab)+1, len(reports_dict)))
        i = 0
        for pkey, word_list in reports_dict.items():
            col = make_unary_feature_vector(vocab, word_list)
            col_w_pkey = np.insert(col, 0, [pkey], 0)
            feature_matrix[:, i] = col_w_pkey[:,0]
            if pkey in pos_pkeys:
                labels[0, i] = 1
            else: 
                labels[0, i] = -1
            i += 1
        return (feature_matrix, labels)
    
