import aws_recog as aws
import pickle
import perceptron as ml
import numpy as np

from PIL import Image

def save_feature_vectors(featDict, filename="./features.p"):
    pickle.dump(featDict, open(filename, "wb"))

def load_feature_vectors(featDict, filename="./features.p"):
    return pickle.load(open(filename, "rb"))

def perf_given_categories(allowed_cat={'Flood':0, 'Water':1, 'Puddle':2, 'Person':3} ):
    '''
    if we ask aws for these categories, what performance do we get? 

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

    # number of reports w/ 'Flood' > .50
    print(np.sum(matrix[0,:] > 0))

    th, th0 = ml.perceptron(matrix, labels)
    print(th, th0)


    score = ml.xval_learning_alg(ml.perceptron, matrix, labels, 5)
    print(score)

    # s = ml.eval_classifier(

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


    ml.plot_separator(ax, th, th0)
    input('go?')

if __name__ == "__main__":
    labels = aws.read_labels_from_disk()
    clean = aws.clean_if_dirty(labels)


    #more = ['Car', 'Traffic Jam', 'Motorcycle']
    allowed = {'Flood':0, 'Water':1, 'Puddle':2, 'Person':3}
    perf_given_categories(allowed)

    #what if we only used the 'Flood' category?
    perf_given_categories({"Flood":0, "noop":1})
