import cv2

def draw_bounding_box(image_path, label_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Leggi le coordinate della bounding box dal file YOLO
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, center_x, center_y, box_width, box_height = map(float, line.split())
        
        # Calcola le coordinate della bounding box
        left = int((center_x - (box_width / 2)) * width)
        top = int((center_y - (box_height / 2)) * height)
        right = int((center_x + (box_width / 2)) * width)
        bottom = int((center_y + (box_height / 2)) * height)
        
        # Disegna la bounding box sull'immagine
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Mostra l'immagine con la bounding box
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Esempio di utilizzo
image_path = r'C:\Users\cesco\OneDrive\Desktop\hollywood_small\train\mov_001_007630.jpeg'
label_path = r'C:\Users\cesco\OneDrive\Desktop\hollywood_small\train\mov_001_007630.txt'
draw_bounding_box(image_path, label_path)
