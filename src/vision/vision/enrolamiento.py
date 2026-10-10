import cv2
import os
import pickle
from insightface.app import FaceAnalysis

def main():
    print("Cargando ArcFace")
    app = FaceAnalysis(name='buffalo_l', root='/ros2_ws/src/vision/models/insightface',
                       allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640,640)) # para uso del cpu

    base_dir = os.path.expanduser('/ros2_ws/src/vision/vision/fotos_equipo')
    
    
    embeddings_db = {}
    for filename in os.listdir(base_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            img = cv2.imread(os.path.join(base_dir, filename))
            faces = app.get(img)

            if len(faces) >0:
                embeddings_db[name] = faces[0].embedding
                print(f"Rostro enrolado: {name}")
            else:
                print(f"Error: No se detecta la cara {filename}")

    db_path = os.path.expanduser('/ros2_ws/src/vision/vision/caras_enroladas.pkl')
    with open(db_path, 'wb') as f:
        pickle.dump(embeddings_db, f)

    print(f"\nBase de datos guardada en {db_path}")

if __name__ == '__main__':
    main()