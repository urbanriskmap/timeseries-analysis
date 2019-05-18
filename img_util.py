import image_recognition.aws_recog as aws
import pickle
import image_recognition.perceptron as ml
import numpy as np

from PIL import Image

def save_feature_vectors(featDict, filename="./features.p"):
    pickle.dump(featDict, open(filename, "wb"))

def load_feature_vectors(featDict, filename="./features.p"):
    return pickle.load(open(filename, "rb"))

def perf_given_categories(allowed_cat={'Flood':0, 'Water':1, 'Puddle':2, 'Person':3} ):
    '''
    if we ask aws for these categories, what performance do we get? 

    returns: 
        Most important labels

    '''
    labels = aws.read_labels_from_disk()
    clean = aws.clean_if_dirty(labels)

    vects = aws.make_feature_vectors(clean, allowed_cat)
    save_feature_vectors(vects)

    l = len(list(allowed_cat))
    # don't include zero data - first row is pkey of data
    matrix_w_pkey = aws.make_matrix_rep(vects, l)
    labels_w_pkey = aws.make_labels_rep(vects)

    matrix = matrix_w_pkey[1:,:]
    labels = labels_w_pkey[1:,:]

    print(np.sum(labels== -1))
    print(np.sum(labels== 1))
    print(np.sum(labels== -1)/labels.shape[1])

    th, th0 = ml.perceptron(matrix, labels, params={"T":1000})
    # print(th, th0)

    # get the indexes of the top n elements in th
    # correspond to the most important params 
    top_n = 30
    max_indicies = np.argpartition(np.abs(th), -top_n, axis=0)[-top_n:][:,0]
    print(max_indicies)

    reverse_allowed = [0]*(len(allowed_cat.keys()) +1)
    for label, index in allowed_cat.items():
        reverse_allowed[index] = label

    print(len(reverse_allowed))
    print(th.shape)
    max_labels = [(reverse_allowed[each], th[each,0]) for each in max_indicies]
    print("MAX LABEL: ", max_labels)
    print("MAX LABEL: ", [each[0] for each in max_labels])
    
    #score = ml.xval_learning_alg(ml.perceptron, matrix, labels, 5)
    #print(score)

    # for allowed_cat = {'Flood':0, 'Flooding':1, 'Water':2, 'Puddle':3, 'Person':4}
    # th = np.array([[-124.51231384], [ 0.], [-163.16449738], [ -45.8591156 ], [ -23.60595703]])
    # th0 = np.array([[-1.]])

    # for flood, water:
    #th = np.array([[-48.12107849], [-93.25553894]]) 
    #th0 = np.array([[-1.]])
    ax = ml.plot_data(matrix, labels, picker=True)

    # now we know each of these points is an image in ./img folder
    # so let's show that image when you hover over a point
    fig = ax.get_figure()

    ax.change_geometry(1, 2, 1)

    # add the new 
    imAx = fig.add_subplot(1, 2, 2) 
    def onpick(event):
        ind = event.ind
        pkeys = [ str(int(matrix_w_pkey[0, each_index])) for each_index in ind]
        print('picked pkeys: ', pkeys)
        print('opening pkeys: ', pkeys[0])
        im = Image.open('./img/'+str(int(matrix_w_pkey[0, ind[0]])) + '.jpeg')
        imAx.imshow(im)
        det = [(k,  np.around(matrix_w_pkey[v+1, ind[0]], 3)) for k,v in allowed_cat.items()]
        imAx.set_title(str(det))
        #imAx.set_title(str(allowed_cat) + '\n' + str(matrix_w_pkey[:, ind[0]]))

    fig.canvas.mpl_connect('pick_event', onpick)


    # don't plot separators that are larger than 2dims...
    if (th.shape[0] <=2):
        ml.plot_separator(ax, th, th0)

    input('go?')
    return max_labels

if __name__ == "__main__":
    labels = aws.read_labels_from_disk()
    clean = aws.clean_if_dirty(labels)

    all_labels = set()
    for key, each in clean.items():
        for lab in each['Labels']:
            if lab['Name'] not in all_labels:
                all_labels.add(lab['Name'])
        
    print(len(all_labels))


    # allowed has label: index pairs for ex: 'Flood':0, 
    # so 'Flood' is the first 
    # in the feature vector 

    # all labels from AWS 
    allowed = dict([ (current_label, index) for index, current_label in enumerate(list(all_labels))])


    # 'common sense labels'
    # more = ['Flood', 'Water', 'Puddle', 'Person', 'Car', 'Traffic Jam', 'Motorcycle']
    # allowed = dict([ (current_label, index) for index, current_label in enumerate(list(more))])

    # top10 =  ['Planter', 'Slum', 'Strap', 'Sedan', 'Grove', 'Parade', 'Ground', 'Vacation', 'Coupe', 'Trail']

    # top30 labels according to perceptron
    # top30 = ['Woodland', 'Man', 'Radiator', 'Sports Car', 'Hand', 'Studio', 'Cottage', 'Beach', 'Reservoir', 'Desk', 'Panoramic', 'Towpath', 'Ripple', 'Countryside', 'Mammal', 'Construction', 'Wildlife', 'Coupe', 'Parade', 'Slum', 'Grove', 'Strap', 'Fish', 'Coffee Table', 'Ground', 'Planter', 'Trail', 'Vacation', 'Sedan', 'Boiling']
    # allowed = dict([ (current_label, index) for index, current_label in enumerate(list(top30))])
    
    perf_given_categories(allowed)

    # what if we only used the 'Flood' category?
    # need to graph on 2 axis so give a no-op label (all zeros)
    # perf_given_categories({"Flood":0, "noop":1})
    # classifies all as negative -> no good

    # plan: get the signed distance of the perceptron separator for each image 
    # take the sigmoid(sd) so that we get a [0,1] range that can be interpreted as a 
    # probability 
    # now throw that into a small dense net to do boosting 
    # along with the sentiment analysis and the flood height, so that 
    # the feature vector is: [ sd of image, negative sentiment, flood height] 
    # then boosting tells us the best mixture of those features
