import chromadb
import os

def main():
    # Base de datos local
    db_path = os.path.expanduser('~/ros2_ws/src/vision/vision/chroma_db')
    client = chromadb.PersistentClient(path=db_path)

    # conocimiento del robot
    coleccion = client.get_or_create_collection(name="robot_knowledge")

    textos = [
        "Proyecto de Candidates 2026",
        "Código: 123",
        "Autor del proyecto: Alex"
    ]

    ids = ["doc1", "doc2", "doc3"]

    # se van a Chroma DB
    coleccion.add(
        documents=textos,
        ids=ids
    )

    print("Se agregaron los textos a la memoria")

if __name__ == "__main__":
    main()