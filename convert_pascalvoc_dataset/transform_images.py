# import cv2
# import os
# from tqdm import tqdm
# import json
# import numpy as np
# from argparse import ArgumentParser
# import math
# import random


# ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
# pixel_border = 40


# def parse_inputs():
#     """ Parser function to take care of the inputs """
#     parser = ArgumentParser(description='Argument: python transform_images.py <data_direction> <output_dir>')
#     parser.add_argument('data_dir', type=str, default="../cmnd_back",
#                         help='Enter path to data direction.')
#     parser.add_argument('output_dir', type=str, default="../cmnd_back_transform",
#                         help='Enter the path of the output of transformation.')
#     args = parser.parse_args()

#     return (args.data_dir, args.output_dir)


# def distance_two_points(p1, p2):
#     return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


# def get_center_point(points):
#     center_point = [0, 0]
#     for point in points:
#         center_point[0] += point[0] / len(points)
#         center_point[1] += point[1] / len(points)
#     return np.array(center_point)


# def adjust_gamma(image, gamma=1.0):
    
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
        
# 	return cv2.LUT(image, table)


# def preprocess_image(image):
#     dst = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
#     dst = cv2.copyMakeBorder(dst, pixel_border, pixel_border, pixel_border, pixel_border, cv2.BORDER_CONSTANT,value=(255,255,255))
#     return dst


# def rotate_box_in_image(corners, angle, width, height, nW, nH):
    
#     center_image = [width//2, height//2]
#     dW = nW - width
#     dH = nH - height

#     rad = angle * math.pi / 180

#     result = []
#     for corner in corners:
#         x_new = center_image[0] + (corner[0] - center_image[0])*math.cos(rad) + (corner[1] - center_image[1])*math.sin(rad) + dW / 2
#         y_new = center_image[1] - (corner[0] - center_image[0])*math.sin(rad) + (corner[1] - center_image[1])*math.cos(rad) + dH / 2
#         result.append([x_new, y_new])
    
#     return result


# def rotate_image(image, angle, gamma):

#     height, width, _ = image.shape
#     image = adjust_gamma(image, gamma)

#     M = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)

#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
    
#     nW = int((height * sin) + (width * cos))
#     nH = int((height * cos) + (width * sin))
    
    
#     M[0, 2] += (nW / 2) - width//2
#     M[1, 2] += (nH / 2) - height//2
#     new_img = cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))

#     return new_img


# def augmentation(image, points, labels):
    
#     height, width, _ = image.shape

#     angles = [random.uniform(-180, 180) for i in range(5)]
#     gammas  = [random.uniform(0.5, 1.7) for i in range(5)]


#     res_images = [image]
#     res_points = [points]
#     res_labels = [labels]

#     for i in range(len(angles)):

#         new_image = rotate_image(image, angles[i], gammas[i])

#         nH, nW, _ = new_image.shape

#         new_points = rotate_box_in_image(points, angles[i], width, height, nW, nH)

#         res_images.append(new_image)
#         res_points.append(new_points)
#         res_labels.append(labels)


#     return res_images, res_points, res_labels



# def transform_images(input_dir, output_dir, augmentation_check = True):

#     if (not os.path.isdir(output_dir)):
#         os.mkdir(output_dir)

#     img_list = []

#     for extension in ALLOWED_EXTENSIONS:
#         img_list.extend([f for f in os.listdir(input_dir) if extension in f])

#     for img_name in tqdm(img_list):

#         img = cv2.imdecode(np.fromfile(os.path.join(input_dir, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)

#         dst = preprocess_image(img)

#         for extension in ALLOWED_EXTENSIONS:
#             if (extension in img_name):
#                 fi = open(os.path.join(input_dir, img_name).replace(extension, ".json"), "r", encoding = "utf-8")

#         data = json.load(fi)


#         annotations = data["shapes"]
#         points = []
#         labels = []
#         for i in range(len(annotations)):
#             point = annotations[i]["points"][0]
#             point[0] += pixel_border
#             point[1] += pixel_border
#             points.append(point)
#             labels.append(annotations[i]["label"])


#         if (augmentation_check):
#             images, new_set_points, set_labels = augmentation(dst, points, labels)

#         for i in range(len(images)):

#             new_points = new_set_points[i]
#             new_labels = set_labels[i]
#             img = images[i]
#             nH, nW, _ = img.shape
#             new_img_name = "{}_".format(i + 1) + img_name
            
#             for k in range(len(data["shapes"])):
#                 label_name = data["shapes"][k]["label"]
#                 data["shapes"][k]["points"][0] = new_points[new_labels.index(label_name)]

#             data["imageHeight"] = nH
#             data["imageWidth"] = nW
#             data["imagePath"] = new_img_name

#             for extension in ALLOWED_EXTENSIONS:
#                 if (extension in img_name):
#                     fo = open(os.path.join(output_dir, new_img_name).replace(extension, ".json"), "w", encoding = "utf-8")
            

#             json.dump(data, fo, indent = 4)
#             fo.close()
            
#             cv2.imwrite(os.path.join(output_dir, new_img_name), img)


# if __name__ == "__main__":
    
#     input_dir, output_dir = parse_inputs()

#     transform_images(input_dir, output_dir, augmentation_check = True)