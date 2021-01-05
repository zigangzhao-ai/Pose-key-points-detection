'''
code by zzg@2020/01/05
'''
#!/usr/bin/python

# pip install lxml
import sys
import os
import json
import glob
import xml.etree.ElementTree as ET
import pdb


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return filename
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


#pdb.set_trace()
def convert(xml_dir, xml_basenames, json_file):
    #list_fp = open(xml_list, 'r')
    list_fp = xml_basenames
    #print(list_fp)
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
       # print(line)
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        # print(xml_f)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': int(image_id)}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            """
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
            """
            keypoints = get_and_check(obj, 'keypoints', 1)
            x1 = float(get_and_check(keypoints, 'x1', 1).text) 
            y1 = float(get_and_check(keypoints, 'y1', 1).text)
            x1f = float(get_and_check(keypoints, 'x1f', 1).text)
            x2 = float(get_and_check(keypoints, 'x2', 1).text)
            y2 = float(get_and_check(keypoints, 'y2', 1).text)
            x2f = float(get_and_check(keypoints, 'x2f', 1).text)
            x3 = float(get_and_check(keypoints, 'x3', 1).text)
            y3 = float(get_and_check(keypoints, 'y3', 1).text)
            x3f = float(get_and_check(keypoints, 'x3f', 1).text)
            x4 = float(get_and_check(keypoints, 'x4', 1).text) 
            y4 = float(get_and_check(keypoints, 'y4', 1).text)
            x4f = float(get_and_check(keypoints, 'x4f', 1).text)

            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   int(image_id), 'bbox':[xmin, ymin, o_width, o_height],
                   'keypoints':[x1,y1,x1f, x2,y2,x2f, x3,y3,x3f, x4,y4,x4f],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'num_keypoints': 4}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent = 1)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':

    xml_dir = "/workspace/zigangzhao/Pose_IDCard/scripts/all_data_0105/xml_train/"
    xml_list = glob.glob(xml_dir + '/*.xml')

    xml_basenames = []
    for item in xml_list:
        xml_basenames.append(os.path.basename(item))

    # f = open('xml_list.txt','w')
    # for xml in xml_basenames:
    #     f.write('\n'+xml)
    json_path = 'json'
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    json_file = "json/train.json"
    convert(xml_dir, xml_basenames, json_file)

    print("finished!")


