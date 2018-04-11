#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import csv
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import time
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

#############################################################################        IOU
def IOU(x1, y1, width1, height1, x2, y2, width2, height2):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。·
    """
    #x1 = Reframe[0];
    #y1 = Reframe[1];
    #width1 = Reframe[2]-Reframe[0];
    #height1 = Reframe[3]-Reframe[1];

    #x2 = GTframe[0];
    #y2 = GTframe[1];
    #width2 = GTframe[2]-GTframe[0];
    #height2 = GTframe[3]-GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width*height; # 两矩形相交面积
        Area1 = width1*height1; 
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    # return IOU
    return ratio #,Reframe,GTframe
##################################################################33#######        IOU

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.4, topn=22):  #0.5, 5
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        #print image_file
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):   # the max number of bbox is topn, so it should be adjusted according to particular dataset
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    file_name_list = os.listdir(args.image_file)
    #统计
    predict_num = 0
    gt_num = 0
    true_positive = 0
    iou_thresh = 0.2
    t = 0
    
    print 'detections begin'
    for i in range(0, len(file_name_list)):
        start = time.time()
        result = detection.detect(args.image_file+file_name_list[i])
        #print result
        end = time.time()
        t = (end -start)
        print ('Detection took {:.3f}s for ' + file_name_list[i]).format(t)
        #print "time of", file_name_list[i], ': ', t
        
        img = Image.open(args.image_file+file_name_list[i])
        draw = ImageDraw.Draw(img)
        width, height = img.size
        #print width, height
        
############################################################################################################
        path = r'examples/images/annotations-csv/'+file_name_list[i].rstrip('.png')+'.csv'
        #print 'csv path: ', path
        
        existed_box = [0] * 50
        existed_box_idx = 0
        for item in result:
            predict_num += 1
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))
            #draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            #draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
            #print [center_x, center_y]
            csv_line_num = -1
            iou_arr = [0] * 50
            iou_idx_arr = [0] * 50
            flag = 0
            out = open(path, 'r')
            read_csv = csv.reader(out, dialect='excel')  #读取标签数据  gt
            for line in read_csv:
                if csv_line_num == -1:
                    csv_line_num += 1
                    continue
                xminT = (int)(float(line[1])) if (int)(float(line[1])) > 1 else 1
                yminT = (int)(float(line[2])) if (int)(float(line[2])) > 1 else 1
                xmaxT = (int)(float(line[1]) + float(line[3])) if (int)(float(line[1]) + float(line[3])) < 500 else 500
                ymaxT = (int)(float(line[2]) + float(line[4])) if (int)(float(line[2]) + float(line[4])) < 500 else 500
                iou_arr[csv_line_num] = IOU(xmin, ymin, xmax-xmin, ymax-ymin, xminT, yminT, xmaxT-xminT, ymaxT-yminT)
                iou_idx_arr[csv_line_num] = csv_line_num
                csv_line_num += 1
            #print iou_arr
            for j in range(0, existed_box_idx):
                del_idx = iou_idx_arr.index(existed_box[j])
                del(iou_arr[del_idx])
                del(iou_idx_arr[del_idx])
            #print [xmin, ymin]
            #print iou_arr
            max_value_idx = iou_arr.index(max(iou_arr))
            if max(iou_arr) > iou_thresh:
                flag = 1
                existed_box[existed_box_idx] = iou_idx_arr[max_value_idx]
                existed_box_idx += 1
            if flag == 1:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0))  #true positive
                true_positive += 1
            else:
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))   #false positive
                #draw.text([xmin, ymin], item[-1] + str(item[-2]), (255, 255, 255))
        #print "detection: ", existed_box
        existed_box = [0] * 50
        existed_box_idx = 0
        csv_line_num = 0
        flag = 0
        out = open(path, 'r')
        read_csv = csv.reader(out, dialect='excel')
        for line in read_csv:
            if csv_line_num == 0:
                csv_line_num += 1
                continue 
            gt_num += 1
            xminT = (int)(float(line[1])) if (int)(float(line[1])) > 1 else 1
            yminT = (int)(float(line[2])) if (int)(float(line[2])) > 1 else 1
            xmaxT = (int)(float(line[1]) + float(line[3])) if (int)(float(line[1]) + float(line[3])) < 500 else 500
            ymaxT = (int)(float(line[2]) + float(line[4])) if (int)(float(line[2]) + float(line[4])) < 500 else 500
            #print [xminT, yminT]
            result_box_idx = 0
            iou_arr = [0] * 50
            iou_idx_arr = [0] * 50
            for item in result:
                xmin = int(round(item[0] * width))
                ymin = int(round(item[1] * height))
                xmax = int(round(item[2] * width))
                ymax = int(round(item[3] * height))
                iou_arr[result_box_idx] = IOU(xmin, ymin, xmax-xmin, ymax-ymin, xminT, yminT, xmaxT-xminT, ymaxT-yminT)
                iou_idx_arr[result_box_idx] = result_box_idx
                #print [xmin, ymin]
                #print iou_arr[result_box_idx]
                result_box_idx += 1
            #print iou_arr
            for j in range(0, existed_box_idx):
                del_idx = iou_idx_arr.index(existed_box[j])
                del(iou_arr[del_idx])
                del(iou_idx_arr[del_idx])
            max_value_idx = iou_arr.index(max(iou_arr))
            #print max(iou_arr)
            if max(iou_arr) > iou_thresh:
                flag = 1
                existed_box[existed_box_idx] = iou_idx_arr[max_value_idx]
                existed_box_idx += 1    
            if flag == 0:
                draw.rectangle([xminT, yminT, xmaxT, ymaxT], outline=(0, 0, 255))  # false negative
            flag = 0 
        out.close()
        #print "gt: ", existed_box
#########################################################################################################          
            #draw.text([xmin, ymin], item[-1] + str(item[-2]), (0, 0, 255))
            #print item
            #print [xmin, ymin, xmax, ymax]
            #print [xmin, ymin], item[-1]
        img.save('detect_results/'+file_name_list[i])
        #print "saved ", file_name_list[i]
    print 'gt_num = ', gt_num
    print 'predict_num = ', predict_num
    print 'true_positive = ', true_positive
    P = 1000 * true_positive / predict_num
    R = 1000 * true_positive / gt_num
    print 'precision = ', P / 10.
    print 'recall = ', R / 10.
    print 'F1 = ', 2 * P * R / (P+R) / 10.


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='data/VOC2007/labelmap_voc.prototxt')
    parser.add_argument('--model_def',
                        default='models/ZFNet/VOC2007/SSDzf_400x400/deploy.prototxt')
    parser.add_argument('--image_resize', default=400, type=int)
    parser.add_argument('--model_weights',
                        default='models/ZFNet/VOC2007/SSDzf_400x400/ZF_VOC2007_SSDzf_400x400_iter_20000.caffemodel')
    parser.add_argument('--image_file', default='examples/images/test_images/')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
