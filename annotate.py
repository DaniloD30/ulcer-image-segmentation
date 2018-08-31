import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

data = "data/"
annotations = "annotations_xml/"
result = "annotations_images/"

# Pegar cada xml de /annotations
# Procurar seu respectivo jpg em /data
# Iniciar um array do numpy com 0 e o mesmo size do jpg
# Encontrar pt/x e pt/y em cada polygon do xml
# Inserir essa informação num array do polígono
# Chamar fillConvexPoly com o array do polígono para desenhar no array do numpy (que é a imagem final)

for image in os.listdir(data):

    zero_img = np.zeros(cv2.imread(data + image).shape, dtype=np.int32)

    annotation = annotations + image.split(".")[0] + ".xml"

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

            cv2.fillConvexPoly(zero_img, np.array(pts, dtype=np.int32), (255, 0, 0))

    cv2.imwrite(result + image, zero_img)
