import aws_recog as aws
import pickle
import perceptron as ml
import numpy as np

from PIL import Image

def save_feature_vectors(featDict):
    pickle.dump(featDict, open("./features.p", "wb"))

def load_feature_vectors(featDict):
    return pickle.load(open("./features.p", "rb"))


if __name__ == "__main__":
    labels = aws.read_labels_from_disk()
    clean = aws.clean_if_dirty(labels)


    # allowed = {'Flood':0, 'Water':1}
    allowed = {'Flood':0, 'Water':1, 'Puddle':2, 'Person':3}
    vects = aws.make_feature_vectors(clean, allowed)
    save_feature_vectors(vects)

    l = len(list(allowed))
    # don't include zero data - first row is pkey of data
    matrix_w_pkey = aws.make_matrix_rep(vects, l)
    labels_w_pkey = aws.make_labels_rep(vects)

    matrix = matrix_w_pkey[1:,:]
    labels = labels_w_pkey[1:,:]
    print(labels)

    # use zeros for unknown pkey
    #matrix = aws.make_matrix_rep_zeros(vects, l)
    #labels = aws.make_labels_rep_zeros(vects)

    th, th0 = ml.perceptron(matrix, labels)
    print(th, th0)


    # score = ml.xval_learning_alg(ml.perceptron, matrix, labels, 10)
    # print(score)

    # for allowed = {'Flood':0, 'Flooding':1, 'Water':2, 'Puddle':3, 'Person':4}
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
        print('picked pkeys: ', ind)
        im = Image.open('./img/'+str(int(matrix_w_pkey[0, ind[0]])) + '.jpeg')
        imAx.imshow(im)
        det = [(k,  matrix_w_pkey[v+1, ind[0]]) for k,v in allowed.items()]
        imAx.set_title(str(det))
        #imAx.set_title(str(allowed) + '\n' + str(matrix_w_pkey[:, ind[0]]))

    fig.canvas.mpl_connect('pick_event', onpick)


    # ml.plot_separator(ax, th, th0)
    input('go?')
    



# data is dirty look at pkley = 76513 
