import pandas as pd
from sqlalchemy import create_engine

global engine
DATABASE = "cognicity"
ID_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)

CH_DATABASE = "riskmap"
CH_ENGINE = create_engine("postgresql://postgres:postgres@localhost:5432/"+ CH_DATABASE)

def get_image_urls():
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


def fetch_images(location='id'):
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
