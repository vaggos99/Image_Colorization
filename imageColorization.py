import sys
import numpy as np
import cv2
from sklearn import cluster
from skimage.segmentation import  slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn import svm,metrics
import matplotlib.pyplot as plt

def quantize(raster, n_colors):
    (h, w) = raster.shape[:2]
# reshape the image into a feature vector so that k-means
# can be applied
    raster = raster.reshape((raster.shape[0] * raster.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
    clt = cluster.MiniBatchKMeans(n_clusters = n_colors)
    #take label of every color
    labels = clt.fit_predict(raster)
    #take the lab colors 
    lab_colors=clt.cluster_centers_.astype("uint8")
    quant = clt.cluster_centers_.astype("uint8")[labels]
# reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    raster = raster.reshape((h, w, 3))

# display the images and wait for a keypress
    cv2.imshow("reff_lab and reff_quant_lab", np.hstack([raster, quant]))
    
    return quant,lab_colors,clt
    


def slic_with_skimage_source(img):
    segments = slic(img_as_float(img), n_segments = 100, sigma = 5)
    superpixels=[]
    for (i, segVal) in enumerate(np.unique(segments)):
       mask = np.zeros(img.shape[:2], dtype = "uint8")
       mask[segments == segVal] = 255
        # show the masked region
       superpixel=cv2.bitwise_and(img, img, mask = mask)
       superpixels.append(superpixel)
    # show the output of SLIC
    fig = plt.figure("Superpixels source")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(img), segments))
    plt.axis("off")
    plt.show()
    return segments,superpixels
    
def slic_with_skimage_target(img):
    segments = slic(img_as_float(img),  n_segments=100, compactness=0.1, sigma=1)
    superpixels=[]
    for (i, segVal) in enumerate(np.unique(segments)):
       mask = np.zeros(img.shape[:2], dtype = "uint8")
       mask[segments == segVal] = 255
        # show the masked region
       superpixel=cv2.bitwise_and(img, img, mask = mask)
       superpixels.append(superpixel)
    # show the output of SLIC
    fig = plt.figure("Superpixels target")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(img), segments))
    plt.axis("off")
    plt.show()
    return segments,superpixels
    
def getGabor(img, ksize, sigma, theta, lamda, gamma, l, ktype):

    kernel=cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, l, ktype=ktype)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    filteredImage=fimg.reshape(-1)

    return filteredImage

def getGabors(img,segments):
    gabor_features=[]
    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(img.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        # show the masked region
        superpixel=cv2.bitwise_and(img, img, mask = mask)
        ksize=5
        thetas = list(map(lambda x: x*np.pi*0.25, [1, 2]))
        gabors=[]
        for theta in thetas:
            for sigma in (1,3):
                for lamda in np.arange(0, np.pi, np.pi/5):
                    for gamma in (0.05, 0.5):
                        gabor = getGabor(superpixel.reshape(-1), ksize, sigma, theta, lamda, gamma, 0, cv2.CV_32F)  
                        gabors.append(np.mean(gabor))
        gabor_features.append(gabors)
        
    return gabor_features
    

def surf_features(img,segments):
    surf = cv2.xfeatures2d.SURF_create()
    surf.setExtended(True)
    surf_features=[]
    for (i, segVal) in enumerate(np.unique(segments)):
        
        # construct a mask for the segment
       
        mask = np.zeros(img.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        # show the masked region
        superpixel=cv2.bitwise_and(img, img, mask = mask)
        #cv2.imshow("Applied", cv2.bitwise_and(img, img, mask = mask))
        #cv2.imshow("surf",cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 255)))
        surf = cv2.xfeatures2d.SURF_create()
        keypoints ,descriptors= surf.detectAndCompute(superpixel, None)
        surf_features.append(np.mean(descriptors, axis=0).tolist())
    return surf_features

def orgazize_source_data(labels,reff_superpixels,surf_f,gabors_f,kmeans):
    #keep ab colors
    labels_ab=labels[:,1:]
   
    labels_ab_dict={}
    # Keep a LAB to index dictionary for all quantized colors
    for idx, color in enumerate(labels_ab):
        labels_ab_dict[color[0], color[1]] = idx
   

    centroid_colors = []
    for superpixel in reff_superpixels:
        # Find all nonzero pixels within the superpixel
        x_s, y_s, _ = np.nonzero(superpixel)
        items = [superpixel[i, j, :] for i, j in zip(x_s, y_s)]
        items = np.array(items)
       
        # Calculate the mean of L, a, b values
        avg_L = np.mean(items[:, 0])
        avg_a = np.mean(items[:, 1])
        avg_b = np.mean(items[:, 2])

        # Quantized the mean color of the superpixel using k-means
        label = kmeans.predict([[avg_L, avg_a, avg_b]])
        
        # Store a, b values of the superpixel
        color = labels[label, 1:]
        #print(color)
       
        centroid_colors.append(color)
        
    features=[]
    aim=[]

    for i in range(len(reff_superpixels)):
        # For each superpixel get the surf, gabor values and the color
        color = centroid_colors[i]

        sample =  surf_f[i]+gabors_f[i]
        features.append(sample)
        #y is the index of the ab color in the dictionary which is the index in the array labels_ab
        aim.append(labels_ab_dict[color[0, 0], color[0, 1]])
    return features,aim,labels_ab
        
def organize_target_data(target_superpixels,surf_f,gabors_f):  
    features=[]  
    for i in range(len(target_superpixels)):
        # For each superpixel get the surf and gabor values
        sample =  surf_f[i]+gabors_f[i]
        features.append(sample)
    
    return features
         
def train_svm(x,y):
   

        # Create a new SVM and train it on source data and labels
    classifier = svm.SVC()
    classifier.fit(x, y)

    # Make predictions and compute accuracy
    predictions = classifier.predict(x)
    print("training accuracy score:",metrics.accuracy_score(y, predictions))
    return classifier

def image_colorization(img,target_superpixels,x,labels,classifier):
        

    # Get predicted labels using the SVM for the target dataset
    l = classifier.predict(x)

        # Get a, b values for color
    color_labels = labels[l]

    #print(l)

  

    # Create a blank copy of the target image to colorize
    colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')

    for i, superpixel in enumerate(target_superpixels):
        # For each superpixel find nonzero pixels
        x_s, y_s = np.nonzero(superpixel)
      
        for k, j in zip(x_s, y_s):
                # Colorize every pixel according to the predicted values
            L = img[k, j]
            a = color_labels[i, 0]
            b = color_labels[i, 1]

            colored_img[k, j, 0] = L
            colored_img[k, j, 1] = a
            colored_img[k, j, 2] = b

     # Convert the colorized image from LAB to RGB and store it
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_LAB2BGR)
    return colored_img

             

    
        




np.set_printoptions(threshold=sys.maxsize)
reff_img = cv2.imread('coloredCastle.jpg',cv2.IMREAD_COLOR)
target_img = cv2.imread('targetCastle.jpg',cv2.IMREAD_GRAYSCALE)
reff_img = cv2.cvtColor(reff_img, cv2.COLOR_BGR2LAB)


print('quantize colors')
reff_quant_img,labels,kmeans=quantize(reff_img,8)
print('extracting superpixels for source')
reff_segments,reff_superpixels=slic_with_skimage_source(reff_quant_img)
print('extracting superpixels for target')
target_segmnents,target_superpixels=slic_with_skimage_target(target_img)
print('extracting surf features from source')
surf_f_source=surf_features(reff_quant_img,reff_segments)
print('extracting surf features from target')
surf_f_target=surf_features(target_img,target_segmnents)
print('extracting gabor features from source')
gabors_f_source=getGabors(reff_quant_img[:,:,0],reff_segments)
print('extracting gabor features from target')
gabors_f_target=getGabors(target_img,target_segmnents)
print('organizing data for train')
features_train,aim_train,labels_ab=orgazize_source_data(labels,reff_superpixels,surf_f_source,gabors_f_source,kmeans)
print('organizing data for target') 
features_target=organize_target_data(target_superpixels,surf_f_target,gabors_f_target)
 
print('training SVM')
classifier=train_svm(features_train,aim_train)
colored_img=image_colorization(target_img,target_superpixels,features_target,labels_ab,classifier)

cv2.imshow('gray image',target_img)
cv2.imshow('colored image',colored_img)
cv2.waitKey(0)
cv2.destroyAllWindows()














