import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

data = "data/"
annotations_data = "annotations_xml_data/"
result_dir_data = "annotations_images_data/"

test = "test/"
annotations_test = "annotations_xml_test/"
result_dir_test = "annotations_images_test/"

# Pegar cada xml de /annotations
# Procurar seu respectivo jpg em /data
# Iniciar um array do numpy com 0 e o mesmo size do jpg
# Encontrar pt/x e pt/y em cada polygon do xml
# Inserir essa informação num array do polígono
# Chamar fillConvexPoly com o array do polígono para desenhar no array do numpy (que é a imagem final)

def annotate(d, a, r):

    for image in os.listdir(d):

        zero_img = np.zeros(cv2.imread(d + image).shape, dtype=np.int32)

        annotation = a + image.split(".")[0] + ".xml"

        if not os.path.isfile(annotation):

            continue

        print("Setting for " + image + " ...")

        root = ET.parse(annotation).getroot()

        objects = root.findall("object")

        for obj in objects:

            polygons = obj.findall("polygon")

            for polygon in polygons:

                pts = []

                for point in polygon.findall("pt"):

                    x = int(point[0].text)
                    y = int(point[1].text)

                    pts.append([x, y])

                cv2.fillConvexPoly(zero_img, np.array(pts, dtype=np.int32), (1, 1, 1))

        zero_img = cv2.resize(zero_img, (480, 360), interpolation=0)

        cv2.imwrite(r + image, zero_img)

        original_img = cv2.imread(d + image)

        original_img = cv2.resize(original_img, (480, 360), interpolation=0)

        cv2.imwrite(d + image, original_img)

annotate(data, annotations_data, result_dir_data)
annotate(test, annotations_test, result_dir_test)
