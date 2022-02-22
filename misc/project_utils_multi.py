import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import pandas as pd
from shutil import copyfile
import json
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras.backend as K
import tensorflow as tf
import keras

ORIGINAL_IMAGE_WIDTH = 960
ORIGINAL_IMAGE_HEIGHT = 720


#using flow_from_dataframe, what do i need?
#original images in rgb, images can be in original size and use the target size parameter
#annotations in 0 to 1 float32 values, these values can be resized to any image by multiplying length and width of target image
#data generator can create train/val sets, so a directory of train data is needed with all images
#augmented dir contains only augmented files, file are flipped horizontal, then original and horizontal are augmented
#augmented annotations are between 0 and 1
#train directory will have subdirectory for each iteration of training
#train directory containing all original + flipped and augmented files, annotations between 0 and 1
#test directory will contain videos, test will be human evaluated, video frames removed using cv2
#how to divide train directory into first, second, third? first_train, second_train, third_train

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, y_cols, directory=None, batch_size=32, dim=(224, 224), n_channels=3, shuffle=True, seed=None):
        self.df = df.reset_index(drop=True)
        self.y_cols = y_cols
        self.directory = directory
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.df.shape[0] / self.batch_size))

    def __getitem__(self, index):
        indexes = self.df.index.tolist()[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1.0, random_state=self.seed)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        for i, index in enumerate(indexes):
            if self.directory is not None:
                file_path = self.directory + self.df.iloc[index]['id']
            else:
                file_path = self.df.iloc[index]['id']
            image = cv2.imread(file_path)
            image = cv2.resize(image, self.dim)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            X[i,] = image
        y = self.df.iloc[indexes,:][self.y_cols].to_numpy()
        return X, [y[:,0], y[:,1:]]
    
    def get_length(self):
        return self.df.shape[0]
    
    def get_entry(self, index):
        if index >= self.get_length():
            return None, None, None
        X, y = self.__data_generation([index])
        return X[0].copy(), y[0][0].copy(), y[1][0].copy()

def create_data_generators(df, y_cols, directory=None, test_split=None, seed=None):
    if test_split == None:
        return DataGenerator(df, y_cols, directory=directory)
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return DataGenerator(train, y_cols, directory=directory), DataGenerator(test, y_cols, directory=directory)
    


#Loads images from a directory
#If a json file is passed, loads those positive images
#If no json is passed, loads negative examples
def load_original_data(directory, json_file):
    images = []
    confidence_targets = []
    bounding_box_targets = []
    if json_file is not None:
        json_file = open(directory + json_file)
        json_data = json.load(json_file)
        for entry in json_data:
            #load annotations, annotations are center of box, convert to x1,y1 and width, height
            width = entry['annotations'][0]['coordinates']['width']
            height = entry['annotations'][0]['coordinates']['height']
            x = entry['annotations'][0]['coordinates']['x'] - width / 2
            y = entry['annotations'][0]['coordinates']['y'] - height / 2
            file_path = directory + entry['image']
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            if width == 0 and height == 0 and x == 0 and y == 0:
                confidence_targets.append(0)
            else:
                confidence_targets.append(1.0)
            bounding_box_targets.append([y, x, (y + height), (x + width)])
    else:
        for file_name in os.listdir(directory):
            if file_name.endswith('.jpg'):
                file_path = directory + file_name
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                confidence_targets.append(0)
                bounding_box_targets.append([0, 0, 0, 0]) 
    images = np.array(images, dtype='uint8')
    confidence_targets = np.array(confidence_targets, dtype='uint8')
    bounding_box_targets = np.array(bounding_box_targets, dtype='float32')
    return images, confidence_targets, bounding_box_targets

def play_images_from_directory(directory, json_file=None, num_to_iter=1):
    fig, ax = plt.subplots(figsize=(5,5))
    
    if json_file is not None:
        json_file = open(directory + json_file)
        json_data = json.load(json_file)
        
        annotation_dict = {}
        for entry in json_data:
            width = entry['annotations'][0]['coordinates']['width']
            height = entry['annotations'][0]['coordinates']['height']
            x1 = entry['annotations'][0]['coordinates']['x'] - width / 2
            y1 = entry['annotations'][0]['coordinates']['y'] - height / 2
            x2 = x1 + width
            y2 = y1 + height
            file_name = entry['image']
            confidence = 0 if width == 0 and height == 0 and x1 == 0 and y1 == 0 else 1.0
            annotation_dict[file_name] = [confidence, y1, x1, y2, x2]
    
    animation_images = []
    iter_count = 0
    for file_name in os.listdir(directory):
        if iter_count % num_to_iter != 0:
            iter_count += 1
            continue
        if file_name.endswith('.jpg'):
            image = cv2.imread(directory + file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if json_file is not None:
                annotation = annotation_dict[file_name]
                if annotation[0] == 1.0:
                    y1 = annotation[1]
                    x1 = annotation[2]
                    y2 = annotation[3]
                    x2 = annotation[4]
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            animation_images.append([ax.imshow(image)])
            iter_count += 1
        
    ani = animation.ArtistAnimation(fig, animation_images, interval=100, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

def play_images_from_sequence_datagen(generator, model=None, interval=100, threshold=0.5, num_to_iter=1):
    fig, ax = plt.subplots(figsize=(5, 5))
    animation_images = []
    for i in range(0, generator.get_length(), num_to_iter):
        image, confidence, annotation = generator.get_entry(i)
        image_copy = image.copy() * 255.0
        image_copy = image_copy.astype(np.uint8)
        image_copy = cv2.resize(image_copy, (960, 720))
        if model is not None and is_model_multi(model):
            prediction = model.predict(np.reshape(image, (1, 224, 224, 3)))
            conf_pred = prediction[0][0][0]
            pred_y1 = prediction[1][0][0]
            pred_x1 = prediction[1][0][1]
            pred_y2 = prediction[1][0][2]
            pred_x2 = prediction[1][0][3]
            if conf_pred >= threshold:
                cv2.rectangle(image_copy, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), (0, 255, 0), 2)
            cv2.putText(image_copy, 'Conf: {:.1f}'.format(conf_pred), (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_copy, 'Y1: {:.1f}'.format(pred_y1), (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_copy, 'X1: {:.1f}'.format(pred_x1), (2, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_copy, 'Y2: {:.1f}'.format(pred_y2), (2, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image_copy, 'X2: {:.1f}'.format(pred_x2), (2, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
        if confidence > threshold:
            cv2.rectangle(image_copy, (int(annotation[1]), int(annotation[0])), (int(annotation[3]), int(annotation[2])), (255, 0, 0), 2)
        animation_images.append([ax.imshow(image_copy)])
    
    ani = animation.ArtistAnimation(fig, animation_images, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

#plays train/test images from a ImageDataGenerator that has annotations
def play_images_from_datagen(gen_iterator, model=None, interval=100, threshold=0.5, num_to_iter=1):
    fig, ax = plt.subplots(figsize=(5, 5))
    animation_images = []
    
    #save old settings
    old_shuffle = gen_iterator.shuffle
    old_batch = gen_iterator.batch_size

    #set iterator to loop through each sample in order
    gen_iterator.shuffle = False
    gen_iterator.batch_size = 1

    i = 0
    for _ in range(gen_iterator.n):
        if i == 0:
            x, y = gen_iterator.next()
            i += 1
        else:
            for j in range(num_to_iter):
                x, y = gen_iterator.next()
                i += 1
        if i > gen_iterator.n:
            break
        img_copy = x[0].copy()
        img_copy *= 255
        anno = y[0].copy()
        cv2.rectangle(img_copy, (int(anno[2]), int(anno[1])), (int(anno[4]), int(anno[3])), (255, 0, 0), 1)
        if model:
            prediction = model.predict(x)
            if is_model_multi(model):
                pass
            else:
                annotation = prediction[0]
                confidence = output[0]
            if confidence >= threshold:
                cv2.rectangle(img_copy, (int(annotation[2]), int(annotation[1])), (int(annotation[4]), int(annotation[3])), (0, 255, 0), 1)
            if len(anno) > 0:
                iou_value = get_iou_value(anno, prediction)
                cv2.putText(img_copy, 'IoU: {:.1f}'.format(iou_value), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_copy, 'Conf: {:.1f}'.format(output[0]), (2, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_copy, 'Y1: {:.1f}'.format(output[1]), (2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_copy, 'X1: {:.1f}'.format(output[2]), (2, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_copy, 'Y2: {:.1f}'.format(output[3]), (2, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img_copy, 'X2: {:.1f}'.format(output[4]), (2, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
        animation_images.append([ax.imshow(img_copy.astype('int'))])

    #reset iterator
    gen_iterator.shuffle = old_shuffle
    gen_iterator.batch_size = old_batch
    gen_iterator.reset()
    
     
    ani = animation.ArtistAnimation(fig, animation_images, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

#plays images from a video file that has no annotations
def play_images_from_video(video_path, model, interval=100, threshold=0.5):
    fig, ax = plt.subplots(figsize=(5, 5))
    cap = cv2.VideoCapture(video_path)
    animation_images = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_input = cv2.resize(frame, (224, 224))
            img_input = np.array(img_input, dtype='float32') / 255.0
            
            if model is not None and is_model_multi(model):
                prediction = model.predict(np.reshape(img_input, (1, 224, 224, 3)))
                conf_pred = prediction[0][0][0]
                pred_y1 = prediction[1][0][0]
                pred_x1 = prediction[1][0][1]
                pred_y2 = prediction[1][0][2]
                pred_x2 = prediction[1][0][3]
                if conf_pred >= threshold:
                    cv2.rectangle(frame, (int(pred_x1), int(pred_y1)), (int(pred_x2), int(pred_y2)), (0, 255, 0), 2)
            cv2.putText(frame, 'Conf: {:.1f}'.format(conf_pred), (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Y1: {:.1f}'.format(pred_y1), (2, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'X1: {:.1f}'.format(pred_x1), (2, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Y2: {:.1f}'.format(pred_y2), (2, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'X2: {:.1f}'.format(pred_x2), (2, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2, cv2.LINE_AA)
            animation_images.append([ax.imshow(frame.astype('int'))])
        else:
            break
    cap.release()
    ani = animation.ArtistAnimation(fig, animation_images, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())
        







#Takes a numpy array of unprocessed images/annotations and returns the flipped images/annotations
def get_flipped_images(images, confidence, annotations):
    #images: unprocessed rgb images
    #annotations: unprocessed full integer annotations
    bounding_boxes = []
    for i in range(annotations.shape[0]):
        y1 = annotations[i][0]
        x1 = annotations[i][1]
        y2 = annotations[i][2]
        x2 = annotations[i][3]
        if confidence[i] == 0:
            label = 'none'
        else:
            label = 'hand'
        bounding_boxes.append(BoundingBoxesOnImage([BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=label)], shape=images[i].shape))  
    seq = iaa.Sequential([iaa.Fliplr(1.0)])
    augmented_images, augmented_boxes = seq(images=images, bounding_boxes=bounding_boxes) 
    augmented_annotations = []
    for bb in augmented_boxes:
        y1 = bb.bounding_boxes[0].y1_int
        x1 = bb.bounding_boxes[0].x1_int
        y2 = bb.bounding_boxes[0].y2_int
        x2 = bb.bounding_boxes[0].x2_int
        augmented_annotations.append([y1, x1, y2, x2])     
    return augmented_images, confidence, np.array(augmented_annotations)


#Takes a numpy array of unprocessed images/annotations and returns augmented images/annotations
#augments from imgaug used include: affine, translate percent, rotate, shear, and perspective transform
def get_augmented_images(images, confidence, annotations, num_augments, bounding_percent=0.25):
    #num_augments: number of augmented images to return 
    #bounding_percent : percent of target out of image before it is discarded, default image
    #                   is discarded if 25% of the bounding box is out of the augmented image
    bounding_boxes = []
    for i in range(annotations.shape[0]):
        y1 = annotations[i][0]
        x1 = annotations[i][1]
        y2 = annotations[i][2]
        x2 = annotations[i][3]
        if confidence[i] == 0:
            label = 'none'
        else:
            label = 'hand'
        bounding_boxes.append(BoundingBoxesOnImage([BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=label)], shape=images[i].shape))  
    seq = iaa.Sequential([iaa.Affine(scale=(0.75, 0.95), 
                                     translate_percent=(-0.2, 0.2), 
                                     rotate=(-10, 10), 
                                     shear=(-10, 10), mode='edge'), 
                          iaa.PerspectiveTransform(scale=(0.01, 0.15), mode='replicate')])
    augmented_images = []
    augmented_annotations = []
    augmented_confidence = []
    while len(augmented_images) < num_augments:
        image_aug, bb_aug = seq(images=images, bounding_boxes=bounding_boxes)
        #check if enough of the bounding box is inside the image, keep images where the
        #bounding box is 80% inside the image
        indexes_to_remove = []
        for i in range(image_aug.shape[0]):
            fraction = bb_aug[i].bounding_boxes[0].compute_out_of_image_fraction(image_aug[i])
            if fraction > bounding_percent:
                indexes_to_remove.append(i)
        image_aug = np.delete(image_aug, indexes_to_remove, 0)
        for i in sorted(indexes_to_remove, reverse=True):
            del bb_aug[i]
            
        for i in range(0, image_aug.shape[0]):
            y1 = bb_aug[i].bounding_boxes[0].y1_int
            x1 = bb_aug[i].bounding_boxes[0].x1_int
            y2 = bb_aug[i].bounding_boxes[0].y2_int
            x2 = bb_aug[i].bounding_boxes[0].x2_int
            augmented_images.append(image_aug[i])
            augmented_confidence.append(1.0 if bb_aug[i].bounding_boxes[0].label == 'hand' else 0)
            augmented_annotations.append([y1, x1, y2, x2])
            if len(augmented_images) >= num_augments:
                break
    return np.array(augmented_images), np.array(augmented_confidence), np.array(augmented_annotations)
    
    
def save_augmented_images(images, confidence, annotations, directory, json_file=None):
    #images passed in should all be of the same label, hand or no hand
    assert all(x == confidence[0] for x in confidence)
    assert images.shape[0] == annotations.shape[0]
    assert images.shape[0] == confidence.shape[0]
    
    json_data = []
    for i in range(0, images.shape[0]):
        filename = str(i+1).zfill(5) + '.jpg'
        y1 = annotations[i][0]
        x1 = annotations[i][1]
        y2 = annotations[i][2]
        x2 = annotations[i][3]
        width = x2 - x1
        height = y2 - y1
        x = x1 + width / 2
        y = y1 + height / 2
        #if image is positive, create an entry for json file for annotation
        if confidence[i] == 1.0:
            json_data.append({
                'image' : filename,
                'annotations' : [
                    {
                        'label' : 'hand',
                        'coordinates' : {
                            'x' : float(x),
                            'y' : float(y),
                            'width' : float(width),
                            'height' : float(height)
                        }
                    }
                ]
            })
        img = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(directory + filename, img)
    
    if json_file is not None:
        json_dict = json.dumps(json_data)
        f = open(directory + json_file, 'w')
        f.write(json_dict)
        f.close()
    print('Saving complete.')
            
        
def create_augmented_images(directory_list, json_list, save_directory, num_augmented, bounding_percentage=0.25):
    #bounding_percentage : percent of augmented bounding box outside of the image before its thrown out
    def generate_augmented_image(image, annotation):
        success = False
        while success == False:
            bounding_box = BoundingBoxesOnImage([BoundingBox(x1=annotation[2], 
                                                             x2=annotation[4], 
                                                             y1=annotation[1], 
                                                             y2=annotation[3], 
                                                             label='hand' if annotation[0] == 1.0 else 0)], 
                                                shape=image.shape)
            seq = iaa.Sequential([iaa.Affine(scale=(0.75, 0.95), 
                                             translate_percent=(-0.2, 0.2), 
                                             rotate=(-10, 10), 
                                             shear=(-10, 10), 
                                             mode='edge'), 
                                  iaa.PerspectiveTransform(scale=(0.01, 0.15), mode='replicate')])
            image_aug, bb_aug = seq(images=[image], bounding_boxes=bounding_box)    
            fraction = bb_aug.bounding_boxes[0].compute_out_of_image_fraction(image_aug[0])
            if fraction <= bounding_percentage:
                #bounding box is inside image
                success = True
                bb = bb_aug.bounding_boxes[0].clip_out_of_image(image_aug[0])
                y1 = bb.y1_int
                x1 = bb.x1_int
                y2 = bb.y2_int
                x2 = bb.x2_int
                augmented_image = image_aug[0]
                if bb_aug.bounding_boxes[0].label == 'hand':
                    augmented_annotation = [1.0, y1, x1, y2, x2]
                else:
                    augmented_annotation = [0, 0, 0, 0, 0]
        return augmented_image, augmented_annotation
    
    def get_annotation(entry):
        width = entry['annotations'][0]['coordinates']['width']
        height = entry['annotations'][0]['coordinates']['height']
        x = entry['annotations'][0]['coordinates']['x'] - width / 2
        y = entry['annotations'][0]['coordinates']['y'] - height / 2
        if width == 0 and height == 0:
            confidence = 0
        else:
            confidence = 1
        return [confidence, y, x, y + height, x + width]
    
    def get_json_entry(json_file_name, json_annotation):
        width = json_annotation[4] - json_annotation[2]
        height = json_annotation[3] - json_annotation[1]
        x = json_annotation[2] + width / 2
        y = json_annotation[1] + height / 2
        return {'image' : json_file_name, 
                'annotations' : [{'label' : 'hand', 
                                  'coordinates' : {'x' : float(x), 
                                                   'y' : float(y), 
                                                   'width' : float(width), 
                                                   'height' : float(height)}
                                 }]
               }
    
    json_annotation_file = []
    augment_count = 0
    while augment_count < num_augmented:
        current_count = 0
        for i, directory in enumerate(directory_list):
            print('Augmenting directory: ', directory)
            if augment_count >= num_augmented:
                print(' Number augmented: ', current_count)
                break
            image = None
            annotation = None
            if json_list[i] is not None:
                #files are positive examples with a json file containing annotations
                json_file = open(directory + json_list[i])
                json_data = json.load(json_file)
                for entry in json_data:
                    file_name = entry['image']
                    image = cv2.imread(directory + file_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    annotation = get_annotation(entry)
                    augmented_image, augmented_annotation = generate_augmented_image(image, annotation)
                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                    save_name = str(augment_count + 1).zfill(5) + '.jpg'
                    cv2.imwrite(save_directory + save_name, augmented_image)
                    json_annotation_file.append(get_json_entry(save_name, augmented_annotation))
                    augment_count += 1
                    current_count += 1
                    if augment_count >= num_augmented:
#                         print(' Number augmented: ', current_count)
#                         current_count = 0
                        break
            else:
                #files are negative examples with no json file
                for file_name in os.listdir(directory):
                    if file_name.endswith('.jpg'):
                        image = cv2.imread(directory + file_name)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        annotation = [0, 0, 0, 0, 0]
                        augmented_image, augmented_annotation = generate_augmented_image(image, annotation)
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        save_name = str(augment_count + 1).zfill(5) + '.jpg'
                        cv2.imwrite(save_directory + save_name, augmented_image)
                        json_annotation_file.append(get_json_entry(save_name, augmented_annotation))
                        augment_count += 1
                        current_count += 1
                        if augment_count >= num_augmented:
#                             print(' Number augmented: ', current_count)
#                             current_count = 0
                            break
            print(' Number augmented: ', current_count)
            current_count = 0
    assert len(json_annotation_file) > 0
    json_dict = json.dumps(json_annotation_file)
    f = open(save_directory + 'annotations.json', 'w')
    f.write(json_dict)
    f.close()
            
def get_negative_positive_count(directory, json_file):
    assert json_file is not None
    assert directory is not None
    assert os.path.isdir(directory)
    assert os.path.isfile(directory + json_file)
    negative_count = 0
    positive_count = 0
    json_file = open(directory + json_file)
    json_data = json.load(json_file)
    for entry in json_data:
        width = entry['annotations'][0]['coordinates']['width']
        height = entry['annotations'][0]['coordinates']['height']
        x = entry['annotations'][0]['coordinates']['x'] - width / 2
        y = entry['annotations'][0]['coordinates']['y'] - height / 2
        if width == 0 and height == 0 and x == 0 and y == 0:
            negative_count += 1
        else:
            positive_count += 1
    return {'negative' : negative_count, 'positive' : positive_count}         
    

def load_image_dataframes(directory, json_file, scale_data=False):
    assert json_file is not None
    assert os.path.isdir(directory)
    assert os.path.isfile(directory + json_file)
    ORIGINAL_IMAGE_WIDTH = 960
    ORIGINAL_IMAGE_HEIGHT = 720
    file_names = []
    confidence_values = []
    y1_values = []
    x1_values = []
    y2_values = []
    x2_values = []
    
    json_file = open(directory + json_file)
    json_data = json.load(json_file)
    for entry in json_data:
        #load annotations, annotations are center of box, convert to x1,y1 and width, height
        width = entry['annotations'][0]['coordinates']['width']
        height = entry['annotations'][0]['coordinates']['height']
        x = entry['annotations'][0]['coordinates']['x'] - width / 2
        y = entry['annotations'][0]['coordinates']['y'] - height / 2
        file_name = entry['image']
        confidence_value = 1.0
        if width == 0 and height == 0 and x == 0 and y == 0:
            confidence_value = float(0)
        confidence_values.append(confidence_value)
        if scale_data:
            y1_values.append(y / ORIGINAL_IMAGE_HEIGHT)
            x1_values.append(x / ORIGINAL_IMAGE_WIDTH)
            y2_values.append((y + height) / ORIGINAL_IMAGE_HEIGHT)
            x2_values.append((x + width) / ORIGINAL_IMAGE_WIDTH)
        else:
            y1_values.append(y)
            x1_values.append(x)
            y2_values.append(y + height)
            x2_values.append(x + width)
        file_names.append(file_name)
        
    
    df = pd.DataFrame(data={'id' : file_names, 
                            'confidence' : confidence_values, 
                            'y1' : y1_values, 
                            'x1' : x1_values, 
                            'y2' : y2_values, 
                            'x2' : x2_values})
    df['y1'] = df['y1'].apply(np.round)
    df['x1'] = df['x1'].apply(np.round)
    df['y2'] = df['y2'].apply(np.round)
    df['x2'] = df['x2'].apply(np.round)
    for index, row in df.iterrows():
        assert row['confidence'] == 1.0 or row['confidence'] == 0
        assert row['y1'] >= 0 and row['y1'] <= 720
        assert row['x1'] >= 0 and row['x1'] <= 960
        assert row['y2'] >= 0 and row['y2'] <= 720
        assert row['x2'] >= 0 and row['x2'] <= 960

    return df
        
def copy_image_directories(image_directory_list, json_list, copy_directory):
    assert len(image_directory_list) == len(json_list)
    assert os.path.isdir(copy_directory)
    for i in range(len(image_directory_list)):
        assert os.path.isdir(image_directory_list[i])
        if json_list[i] is not None:
            assert os.path.isfile(image_directory_list[i] + json_list[i])
    file_count = 0
    file_names = []
    file_in_paths = []
    file_out_paths = []
    target_values = []
    for i in range(len(image_directory_list)):
        #set of positive examples or negative examples with json file
        if json_list[i] is not None:
            json_file = open(image_directory_list[i] + json_list[i])
            json_data = json.load(json_file)
            for entry in json_data:
                file_name = entry['image']
                width = entry['annotations'][0]['coordinates']['width']
                height = entry['annotations'][0]['coordinates']['height']
                x = entry['annotations'][0]['coordinates']['x']
                y = entry['annotations'][0]['coordinates']['y']
                file_out_name = str(file_count+1).zfill(5) + '.jpg'
                file_names.append(file_out_name)
                file_count += 1
                file_in_paths.append(image_directory_list[i] + file_name)
                file_out_paths.append(copy_directory + file_out_name)
                target_values.append([x, y, width, height])
        #set of negative examples with no json file
        else:
            for file_name in os.listdir(image_directory_list[i]):
                if file_name.endswith('.jpg'):
                    file_out_name = str(file_count+1).zfill(5) + '.jpg'
                    file_names.append(file_out_name)
                    file_count += 1
                    file_in_paths.append(image_directory_list[i] + file_name)
                    file_out_paths.append(copy_directory + file_out_name)
                    target_values.append([0, 0, 0, 0])
    
    assert len(file_in_paths) == len(file_out_paths)
    assert len(file_in_paths) == len(target_values)
    json_data = []
    for i in range(len(target_values)):
        copyfile(file_in_paths[i], file_out_paths[i])
        x = target_values[i][0]
        y = target_values[i][1]
        width = target_values[i][2]
        height = target_values[i][3]
        json_data.append({
                'image' : file_names[i],
                'annotations' : [
                    {
                        'label' : 'hand',
                        'coordinates' : {
                            'x' : x,
                            'y' : y,
                            'width' : width,
                            'height' : height
                        }
                    }
                ]
            })
    assert len(json_data) > 0
    json_dict = json.dumps(json_data)
    f = open(copy_directory + 'annotations.json', 'w')
    f.write(json_dict)
    f.close()


def plot_history(history):
    fig, ax = plt.subplots(2, 3, figsize=(14,10))
    
    ax[0,0].set_title('Overall Training Loss')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].plot(history.history['loss'], label='Training')
    ax[0,0].plot(history.history['val_loss'], label='Validation')
    ax[0,0].legend()
    
    ax[0,1].set_title('Confidence Training Loss')
    ax[0,1].set_xlabel('Epoch')
    ax[0,1].set_ylabel('Loss')
    ax[0,1].plot(history.history['confidence_output_loss'], label='Training')
    ax[0,1].plot(history.history['val_confidence_output_loss'], label='Validation')
    ax[0,1].legend()
    
    ax[0,2].set_title('Bounding Training Loss')
    ax[0,2].set_xlabel('Epoch')
    ax[0,2].set_ylabel('Loss')
    ax[0,2].plot(history.history['bounding_output_loss'], label='Training')
    ax[0,2].plot(history.history['val_bounding_output_loss'], label='Validation')
    ax[0,2].legend()
    
    
    ax[1,0].set_title('Training IOU Metric')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('IOU')
    ax[1,0].set_ylim([0, 1])
    if 'bounding_output_iou_metric_multi' in history.history:
        ax[1,0].plot(history.history['bounding_output_iou_metric_multi'], label='Training')
        ax[1,0].plot(history.history['val_bounding_output_iou_metric_multi'], label='Validation')
    else:
        ax[1,0].plot(history.history['bounding_output_iou__metric'], label='Training')
        ax[1,0].plot(history.history['val_bounding_output_iou__metric'], label='Validation')
    ax[1,0].legend()
    
    ax[1,1].set_title('Training Confidence Accuracy/AUC')
    ax[1,1].set_xlabel('Epoch')
    ax[1,1].set_ylabel('Accuracy')
    ax[1,1].set_ylim([0, 1])
    ax[1,1].plot(history.history['confidence_output_binary_accuracy'], label='Training Accuracy')
    ax[1,1].plot(history.history['val_confidence_output_binary_accuracy'], label='Validation Accuracy')
    ax[1,1].plot(history.history['confidence_output_auc'], label='Training AUC')
    ax[1,1].plot(history.history['val_confidence_output_auc'], label='Validation AUC')
    ax[1,1].legend()
    
    ax[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()

def iou_metric_multi(y_true, y_pred):
    y_true_area = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0]) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1])
    y_pred_area = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0]) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1])

    x1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    y1 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    x2 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])
    y2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    
    intersection = K.maximum(0.0, x2 - x1) * K.maximum(0.0, y2 - y1)
    union = y_true_area + y_pred_area - intersection
    iou = intersection / (union + K.epsilon())
    iou = K.clip(iou, 0, 1.0)
#     print('\nmulti iou: ', iou)
    return iou
        
def iou_metric(y_true, y_pred):
    y_true_area = K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1]) * K.abs(K.transpose(y_true)[4] - K.transpose(y_true)[2])
    y_pred_area = K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1]) * K.abs(K.transpose(y_pred)[4] - K.transpose(y_pred)[2])

    x1 = K.maximum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    y1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    x2 = K.minimum(K.transpose(y_true)[4], K.transpose(y_pred)[4])
    y2 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])
    
    intersection = K.maximum(0.0, x2 - x1) * K.maximum(0.0, y2 - y1)
    union = y_true_area + y_pred_area - intersection
    iou = intersection / (union + K.epsilon())
    iou = K.clip(iou, 0, 1.0)
    return iou

def get_iou_value(y_true, y_pred):
    at = tf.convert_to_tensor(y_true, dtype=tf.float32)
    bt = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    c = iou_metric(at, bt)
    return c.numpy()[0]


def write_results(file_path, model, history, n_training, n_validation, pos_neg_counts=None):
    result_dict = {}
    result_dict['n_training'] = n_training
    result_dict['n_validation'] = n_validation
    result_dict['is_multi'] = True
    if pos_neg_counts is not None:
        result_dict['n_negative'] = pos_neg_counts['negative']
        result_dict['n_positive'] = pos_neg_counts['positive']

    last_layers = []
    for layer in model.get_config()['layers'][23:]:
        if layer['class_name'] == 'Concatenate':
            continue
        layer_dict = {}
        layer_dict['layer_type'] = layer['class_name']
        layer_dict['units'] = layer['config']['units']
        layer_dict['activation'] = layer['config']['activation']
        last_layers.append(layer_dict)
    result_dict['layers'] = last_layers
        
    df_dict = {}
    df_dict['loss'] = history.history['loss']
    df_dict['val_loss'] = history.history['val_loss']
    df_dict['confidence_output_loss'] = history.history['confidence_output_loss']
    df_dict['val_confidence_output_loss'] = history.history['val_confidence_output_loss']
    df_dict['confidence_output_binary_accuracy'] = history.history['confidence_output_binary_accuracy']
    df_dict['val_confidence_output_binary_accuracy'] = history.history['val_confidence_output_binary_accuracy']
    df_dict['confidence_output_auc'] = history.history['confidence_output_auc']
    df_dict['val_confidence_output_auc'] = history.history['val_confidence_output_auc']
    df_dict['bounding_output_loss'] = history.history['bounding_output_loss']
    df_dict['val_bounding_output_loss'] = history.history['val_bounding_output_loss']
    df_dict['bounding_output_iou__metric'] = history.history['bounding_output_iou__metric']
    df_dict['val_bounding_output_iou__metric'] = history.history['val_bounding_output_iou__metric']        
    result_dict['history'] = df_dict
    
    with open(file_path, 'w') as outfile:
        json.dump(result_dict, outfile)
    
def load_results(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def plot_results(results):
    print('Number of Training: ', results['n_training'])
    if 'train_positive' in results:
        print(' Positive: ', results['train_positive'])
    if 'train_negative' in results:
        print(' Negative: ', results['train_negative'])
    print('Number of Validation: ', results['n_validation'])
    if 'val_positive' in results:
        print(' Positive: ', results['val_positive'])
    if 'val_negative' in results:
        print(' Negative: ', results['val_negative'])
    if 'bounding_output_iou_metric_multi' in results['history']:
        print('Best IOU Training: ', max(results['history']['bounding_output_iou_metric_multi']))
        print('Best IOU Validation: ', max(results['history']['val_bounding_output_iou_metric_multi']))
    if 'bounding_output_iou__metric' in results['history']:
        print('Best IOU Training: ', max(results['history']['bounding_output_iou__metric']))
        print('Best IOU Validation: ', max(results['history']['val_bounding_output_iou__metric']))
    print('Best AUC Training: ', max(results['history']['confidence_output_auc']))
    print('Best AUC Validation: ', max(results['history']['val_confidence_output_auc']))
    if 'training_iou' in results:
        print('Final Training IOU: ', results['training_iou'])
    if 'validation_iou' in results:
        print('Final Validation IOU: ', results['validation_iou'])
    if 'testing_iou' in results:
        print('Final Testing IOU: ', results['testing_iou'])
    df = pd.DataFrame(results['layers'])
    display(HTML('<h3>Layers</h3>'))
    display(df)
    history = {'history' : results['history']}
    history = SimpleNamespace(**history)
    plot_history(history)
    
def is_model_multi(model):
    #returns true if model splits confidence / bounding
    #false if its the old type that is a single 5 element tensor
    if model.get_config()['layers'][-2]['config']['units'] == 1 and model.get_config()['layers'][-1]['config']['units'] == 4:
        return True
    return False


#older functions for working with image arrays instead of directory loaders

#Plays a list of original sized unprocessed image files
#If confidence and annotations are included, the bounding boxes are marked.
#If a model is included, the model predicts a bounding box from the original sized image and marks it
def play_original_images(images, confidence=None, annotations=None, model=None, interval=100, threshold=0.5):
    #images: numpy array of unprocessed images
    #confidence: numpy array of 1/0 values
    #annotations: unprocessed annotations
    #model: trained model that takes in processed image
    #interval: number ms between image animations
    #threshold: value of confidence to consider a positive example
    #TODO: add confidence value as text on image
    if confidence is not None:
        assert images.shape[0] == confidence.shape[0]
    if annotations is not None:
        assert images.shape[0] == annotations.shape[0]
    fig, ax = plt.subplots(figsize=(5,5))
    animation_images = []
    for i in range(images.shape[0]):
        img_copy = images[i].copy()
        if model is not None:
            img_input = images[i].copy()
            img_input = cv2.resize(img_input, (224, 224))
            img_input = np.array(img_input, dtype='float32') / 255.0
            prediction = model.predict(np.reshape(img_input, (1, 224, 224, 3)))
            bounding_box = prediction[0]
            y1 = bounding_box[0] * images[i].shape[0]
            x1 = bounding_box[1] * images[i].shape[1]
            y2 = bounding_box[2] * images[i].shape[0]
            x2 = bounding_box[3] * images[i].shape[1]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if annotations is not None:
            if confidence is not None and confidence[i] >= 0.5 or confidence is None:
                y1 = annotations[i][0]
                x1 = annotations[i][1]
                y2 = annotations[i][2]
                x2 = annotations[i][3]
                cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        animation_images.append([ax.imshow(img_copy)])
    ani = animation.ArtistAnimation(fig, animation_images, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

#Takes loaded images/annotations and processes them
#Image pixels are scaled between 0 and 1
#Annotation pixels are scaled between 0 and 1
def process_data(images, annotations):
    #images: numpy array of unprocessed images
    #annotations: numpy array of unprocessed annotations
    processed_images = []
    processed_annotations = np.copy(annotations)
    for i in range(0, images.shape[0]):
        image_height = images[i].shape[0]
        image_width = images[i].shape[1]
        assert (images[i] >= 0).all() and (images[i] <= 255.0).all()
        assert (processed_annotations[i][0] >= 0).all() and (processed_annotations[i][0] <= image_height).all()
        assert (processed_annotations[i][1] >= 0).all() and (processed_annotations[i][1] <= image_width).all()
        assert (processed_annotations[i][2] >= 0).all() and (processed_annotations[i][2] <= image_height).all()
        assert (processed_annotations[i][3] >= 0).all() and (processed_annotations[i][3] <= image_width).all()
        processed_images.append(cv2.resize(images[i], (224, 224)) / 255.0)
        processed_annotations[i][0] /= image_height #y1 / height
        processed_annotations[i][1] /= image_width #x1 / width
        processed_annotations[i][2] /= image_height #y2 / height
        processed_annotations[i][3] /= image_width #x2 / width
    processed_images = np.array(processed_images, dtype='float32')
    assert (processed_images >= 0).all() and (processed_images <= 1.0).all()
    assert (processed_annotations >= 0).all() and (processed_annotations <= 1.0).all()
    return processed_images, processed_annotations

#Plays a list of image files processed to 224x224
#If confidence and annotations are included, the bounding boxes are marked.
#If a model is included, the model predicts a bounding box from the image and marks it
def play_processed_images(images, confidence=None, annotations=None, model=None, interval=100):
    #images: numpy array of processed images
    #confidence: numpy array of 1/0 values
    #annotations: processed annotations
    #model: trained model that takes in processed image
    #interval: number ms between image animations
    assert (images >= 0).all() and (images <= 1.0).all()
    if confidence is not None:
        assert images.shape[0] == confidence.shape[0]
    if annotations is not None:
        assert images.shape[0] == annotations.shape[0]
        assert (annotations >= 0).all() and (annotations <= 1.0).all()
    fig, ax = plt.subplots(figsize=(5,5))
    animation_images = []
    for i in range(images.shape[0]):
        img_copy = images[i].copy() * 255.0
        if model is not None:
            img_input = images[i].copy()
            img_input = cv2.resize(img_input, (224, 224))
            img_input = np.array(img_input, dtype='float32') / 255.0
            prediction = model.predict(np.reshape(img_input, (1, 224, 224, 3)))
            bounding_box = prediction[0]
            y1 = bounding_box[0] * images[i].shape[0]
            x1 = bounding_box[1] * images[i].shape[1]
            y2 = bounding_box[2] * images[i].shape[0]
            x2 = bounding_box[3] * images[i].shape[1]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        if annotations is not None:
            if confidence is not None and confidence[i] >= 0.5 or confidence is None:
                y1 = annotations[i][0] * 224
                x1 = annotations[i][1] * 224
                y2 = annotations[i][2] * 224
                x2 = annotations[i][3] * 224
                cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        animation_images.append([ax.imshow(img_copy.astype('int'))])
    ani = animation.ArtistAnimation(fig, animation_images, interval=interval, blit=True)
    plt.close()
    return HTML(ani.to_html5_video())

#Saves the processed images and annotations to the processed image directory
#Images and annotations are saved in the format ready to load into the model
def save_processed_images(images, annotations, directory):
    #images: numpy array of processed images
    #annotations: numpy array of processed annotations
    assert images.shape[0] == annotations.shape[0]
    assert (images >= 0).all() and (images <= 1.0).all()
    assert (annotations >= 0).all() and (annotations <= 1.0).all()
    annotation_dict = {}
    confidence_dict = {}
    for i in range(images.shape[0]):
        image_name = str(i).zfill(5) + '.npy'
        filepath = directory + image_name
        np.save(filepath, images[i])
        if os.path.exists(filepath):
            annotation = annotations[i]
            confidence = 0.0 if all(annotation == 0) else 1.0
            annotation_dict[image_name] = (confidence, annotation.tolist())
        else:
            print('Error writing {} image to: {}'.format(image_name, filepath))
    json_dict = json.dumps(annotation_dict)
    f = open(directory + 'annotations.json', 'w')
    f.write(json_dict)
    f.close()
    print('Saving complete.')
    

#Loads processed images and annotations into numpy arrays ready for training
def load_processed_data(directory):
    #directory: directory that contains image and/or annotations json file
    images = []
    annotations = []
    confidence = []
    if os.path.exists(directory + 'annotations.json'):
        f = open(directory + 'annotations.json')
        annotation_dict = json.load(f)
        f.close()
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            img = np.load(directory + file)
            images.append(img)
            annotations.append(annotation_dict[file][1])
            confidence.append(annotation_dict[file][0])
    
    return np.array(images, dtype='float32'), np.array(confidence, dtype='float32'), np.array(annotations, dtype='float32')
    
                    
    