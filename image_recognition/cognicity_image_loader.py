import pandas as pd
from sqlalchemy import create_engine

global engine
DATABASE = "cognicity"
ID_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)

CH_DATABASE = "riskmap"
CH_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ CH_DATABASE)

class CognicityImageLoader:

    def __init__(self, databaseEngine):
        ''' Creates a data loader for cognicity data
        Args:
            databaseEngine: a sqlalchemy database engine

        Returns: 
            A cognicity
        '''
        self.database = databaseEngine

    def get_image_urls(self):
        '''
        returns dictionary of {pkey: image_url} for all rows in db that have an image url
        '''
        global CH_ENGINE
        rows = pd.read_sql_query('''
        SELECT pkey, image_url FROM riskmap.all_reports 
            WHERE image_url IS NOT null
            ORDER BY created_at
        ''', con=CH_ENGINE, index_col="pkey")
        
        return rows.to_dict()['image_url']
    
    
    def fetch_images(self, location='id', filename='./img/'):
        image_urls = get_image_urls()
        for key in image_urls.keys():
            each = image_urls[key]
            print("url is: " + each)
            r = requests.get(each, stream=True)
            if r.status_code == 200:
                try:
                    im = Image.open(r.raw)
                    if location == 'id':
                        im.save("./img/"+ str(key) + ".jpeg", "JPEG")
                    else:
                        im.save("./img_ch/"+ str(key) + ".jpeg", "JPEG")
                except:
                    print("ERROR FETCHING", each)
            else: 
                print("ERROR COULD NOT READ URL: " + each)
    
    
    def make_matrix_rep(self, featureDict, lenFeatVect):
    
        # looks like: 
        # pkey0    | pkey1 .. 
        # featvect | featV1
        out = np.zeros((lenFeatVect +1, len(featureDict.keys())))
        for i, pkey in enumerate(sorted(featureDict)):
            l = featureDict[pkey].copy() # shallow copy because they're builtins
            l.insert(0, pkey)
            out[:,i] = np.array(l)
        return out
    
    
    def make_labels_rep(self, featureDict, location='id'):
        # if pkey exists in feature dict, figures out if flooding 
        # else zero
    
        #start_known_flood = "'2017-02-20 00:00:35.630000-05:00'"
        #end_known_flood = "'2017-02-23 00:00:35.630000-05:00'"
        if location == 'id':
            start_known_flood = "2017-02-20 00:00:35.630000-05:00"
            end_known_flood =   "2017-02-23 00:00:35.630000-05:00"
            global engine
            knownFlood = pd.read_sql_query('''
                SELECT pkey from cognicity.all_reports
                WHERE created_at > '2017-02-20 00:00:35.630000-05:00'
                AND created_at < '2017-02-23 00:00:35.630000-05:00'
            ''', con=engine, params={"start_known_flood": start_known_flood, "end_known_flood":end_known_flood})
    
        else:
            start_known_flood = "'2017-11-01 00:00:35.630000-04:00'"
            end_known_flood = "'2017-11-07 00:00:35.630000-04:00'"
    
            knownFlood = pd.read_sql_query('''
                SELECT pkey from riskmap.all_reports
                WHERE created_at > %(start_known_flood)s::timestamptz
                AND created_at < %(end_known_flood)s::timestamptz
            ''', con=CH_ENGINE, params={"start_known_flood": start_known_flood, "end_known_flood":end_known_flood})
    
    
        knownFloodSet = set(knownFlood['pkey'])
        print(knownFloodSet)
    
        out = np.zeros((2, len(featureDict.keys())))
        for i, pkey in enumerate(sorted(featureDict)):
            # look up if this pkey is a flood event
            if pkey in knownFloodSet:
                out[0, i] = pkey
                out[1, i] = 1
            else:
                # no known flooding
                out[0, i] = pkey
                out[1, i] = -1
        return out
