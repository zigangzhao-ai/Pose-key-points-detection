import os,sys
import glob
from PIL import Image
import pdb

# the direction/path of Image,Label
src_img_dir = "/workspace1/zigangzhao/Pose/edge_detect_dataset_generator/datasets/images/"
src_txt_dir = "/workspace1/zigangzhao/Pose/edge_detect_dataset_generator/datasets/annotations/point/"
src_xml_dir = "/workspace1/zigangzhao/test/txt_to_xml/finished02"

img_Lists = glob.glob(src_img_dir + '/*.jpg')
#print(img_Lists)

img_basenames = []
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
    #print(img_basenames)

img_name = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_name.append(temp1)
    #print(img_name)



for img in img_name:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
    #print(width)
    
    
    #gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    #print(gt[0].type())
    #pdb.set_trace()
    
      #+ '/workspace1/zigangzhao/Pose/TF-SimpleHumanPose/data/MPII/images/'
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('<folder>VOC2007</folder>\n')
    xml_file.write('<filename>' +str(img) + '.jpg' + '</filename>\n')
    xml_file.write('<size>\n')
    xml_file.write('<width>' + str(width) + '</width>\n')
    xml_file.write('<height>' + str(height) + '</height>\n')
    xml_file.write('<depth>3</depth>\n')
    xml_file.write('</size>\n')
    #print(len(gt))

    
    
    #for index in range(len(gt)):
        #if index == int(fg)-1 :
    #spt1 = [gt[0]]
    spt1 = gt[0].split(' ')
    spt1 = [ float(x) for x in spt1 ]
    
    #print(spt1[0])
    spt2 = gt[1].split(' ')
    spt2 = [ float(x) for x in spt2 ]
    spt3 = gt[2].split(' ')
    spt3 = [ float(x) for x in spt3 ]
    spt4 = gt[3].split(' ')
    spt4 = [ float(x) for x in spt4 ]
  
       
    #a=[spt1[0],spt2[0],spt3[0],spt4[0]]
    #b=[spt1[1],spt2[1],spt3[1],spt4[1]]
    #x1 = max(a)
    #y1 = max(b)
    if spt1[0] > spt2[0]:
        print("correct!!")
        x1 = spt1[0]
        y1 = spt1[1]
        x2 = spt2[0]
        y2 = spt2[1]
        x3 = spt3[0]
        y3 = spt3[1]
        x4 = spt4[0]
        y4 = spt4[1]


    if spt1[0] < spt2[0]:
        print("exchange!")
        x1 = spt3[0]
        y1 = spt3[1]
        x2 = spt4[0]
        y2 = spt4[1]
        x3 = spt1[0]
        y3 = spt1[1]
        x4 = spt2[0]
        y4 = spt2[1]
            
         
    xml_file.write('<object>\n')
    #xml_file.write('<name>' + str(spt[0]) + '</name>\n')
    xml_file.write('<name>' + 'picture' + '</name>\n')
    xml_file.write('<pose>Unspecified</pose>\n')
    xml_file.write('<truncated>0</truncated>\n')
    xml_file.write('<difficult>0</difficult>\n')
    xml_file.write('<keypoints>\n')
    xml_file.write('<x1>' + str(x1) + '</x1>\n')
    xml_file.write('<y1>' + str(y1) + '</y1>\n')
    xml_file.write('<x1f>' + str(1.0) + '</x1f>\n')
    xml_file.write('<x2>' + str(x2) + '</x2>\n')
    xml_file.write('<y2>' + str(y2) + '</y2>\n')
    xml_file.write('<x2f>' + str(1.0) + '</x2f>\n')
    xml_file.write('<x3>' + str(x3) + '</x3>\n')
    xml_file.write('<y3>' + str(y3) + '</y3>\n')
    xml_file.write('<x3f>' + str(1.0) + '</x3f>\n')
    xml_file.write('<x4>' + str(x4) + '</x4>\n')
    xml_file.write('<y4>' + str(y4) + '</y4>\n')
    xml_file.write('<x4f>' + str(1.0) + '</x4f>\n')
    xml_file.write('</keypoints>\n')
    xml_file.write('</object>\n')         
            
        

    xml_file.write('</annotation>')
print("finished!")
