import cv2



if __name__ == "__main__":
    img = cv2.imread("../ARC_DATAS/arcKG00001_00048/arcKG00001.jpg")    
    img = cv2.resize(img, (256, 256))
