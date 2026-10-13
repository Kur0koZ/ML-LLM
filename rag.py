import os
import json

import mysql.connector
import numpy as np
import ollama

# =========================
# CONFIG
# =========================
MYSQL_HOST = "localhost"
MYSQL_USER = "rag_user"     # Or "root"
MYSQL_PASSWORD = "123456"     # Or ""
MYSQL_DATABASE = "rag_db"

LLM_MODEL = "qwen2.5:3b" #ชื่อโมเดลที่ติดตั้งไว้ในollama
EMBED_MODEL = "bge-m3"   #ชื่อโมเดลที่ติดตั้งไว้ในollama

DOCUMENT_FOLDER = "documents"
TOP_K = 7 


# =========================
# MYSQL CONNECTION
# =========================
def connect_mysql():
    db = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        use_pure=True,
    )
    return db


# =========================
# READ TXT FILE
# =========================
def read_txt(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


# =========================
# CREATE CHUNKS (ตัดตามย่อหน้า: 1 ย่อหน้า = 1 chunk)
# =========================
def create_chunks(text):
    text = text.replace("\r\n", "\n")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# =========================
# CREATE EMBEDDING
# =========================
def create_embedding(text):
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


# =========================
# DELETE OLD VERSION (กันข้อมูลซ้ำเมื่อรันหลายรอบ)
# =========================
def delete_document_if_exists(filename):
    db = connect_mysql()
    cursor = db.cursor()

    cursor.execute(
        "DELETE FROM document_chunks WHERE document_id IN "
        "(SELECT id FROM documents WHERE filename = %s)",
        (filename,),
    )
    cursor.execute("DELETE FROM documents WHERE filename = %s", (filename,))

    db.commit()
    cursor.close()
    db.close()


# =========================
# SAVE DOCUMENT TO MYSQL
# =========================
def save_document_to_mysql(filename, content):
    db = connect_mysql()
    cursor = db.cursor()

    sql = """
    INSERT INTO documents
    (filename, content)
    VALUES (%s, %s)
    """

    cursor.execute(sql, (filename, content))
    document_id = cursor.lastrowid

    db.commit()
    cursor.close()
    db.close()

    return document_id


# =========================
# SAVE CHUNKS TO MYSQL
# =========================
def save_chunks_to_mysql(document_id, chunks):
    db = connect_mysql()
    cursor = db.cursor()

    sql = """
    INSERT INTO document_chunks
    (document_id, chunk_index, chunk_text, embedding)
    VALUES (%s, %s, %s, %s)
    """

    for index, chunk in enumerate(chunks):
        print(f"กำลังสร้าง Embedding {index + 1}/{len(chunks)}")
        embedding = create_embedding(chunk)
        cursor.execute(sql, (document_id, index, chunk, json.dumps(embedding)))

    db.commit()
    cursor.close()
    db.close()


# =========================
# ADD DOCUMENT TO RAG
# =========================
def add_document(filename):
    filepath = os.path.join(DOCUMENT_FOLDER, filename)

    print()
    print("กำลังเพิ่มไฟล์:", filepath)

    text = read_txt(filepath)

    chunks = create_chunks(text)
    print("จำนวน Chunk:", len(chunks))

    # สร้าง embedding ให้ครบก่อน ค่อยลบของเก่า (ถ้า Ollama ล่ม ข้อมูลเดิมจะไม่หาย)
    embeddings = []
    for index, chunk in enumerate(chunks):
        print(f"กำลังสร้าง Embedding {index + 1}/{len(chunks)}")
        embeddings.append(create_embedding(chunk))

    delete_document_if_exists(filename)
    document_id = save_document_to_mysql(filename, text)
    print("บันทึก Document MySQL ID:", document_id)

    db = connect_mysql()
    cursor = db.cursor()
    sql = """
    INSERT INTO document_chunks
    (document_id, chunk_index, chunk_text, embedding)
    VALUES (%s, %s, %s, %s)
    """
    for index, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        cursor.execute(sql, (document_id, index, chunk, json.dumps(emb)))
    db.commit()
    cursor.close()
    db.close()

    print("เพิ่มข้อมูลเข้า RAG สำเร็จ")


# =========================
# SEARCH RAG
# =========================
def search_rag(question, top_k=TOP_K):
    print("\nกำลังค้นหา...")

    q = np.array(create_embedding(question))

    db = connect_mysql()
    cursor = db.cursor()
    cursor.execute(
        "SELECT chunk_text, embedding FROM document_chunks WHERE embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    cursor.close()
    db.close()

    if not rows:
        return []

    texts = [r[0] for r in rows]
    matrix = np.array([json.loads(r[1]) for r in rows])

    scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:top_k]

    return [texts[i] for i in best]


# =========================
# ASK OLLAMA
# =========================
def ask_ollama(question):
    contexts = search_rag(question)

    context_text = "\n\n".join(contexts)

    prompt = f"""
คุณเป็นผู้ช่วย AI

ให้ตอบโดยใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น
ห้ามใช้ความรู้นอกเอกสาร และห้ามเดา

ถ้า Context ไม่มีข้อมูลที่เกี่ยวข้อง
ให้ตอบว่า:

"ไม่พบข้อมูลในเอกสาร"

Context:

{context_text}

คำถาม:

{question}

คำตอบ:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    return response["message"]["content"]


# =========================
# LOAD DOCUMENTS
# =========================
def load_documents():
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)
        print("สร้าง folder documents แล้ว")
        return

    files = os.listdir(DOCUMENT_FOLDER)
    txt_files = [file for file in files if file.endswith(".txt")]

    if len(txt_files) == 0:
        print("ไม่พบไฟล์ .txt")
        return

    for filename in txt_files:
        print()
        print("=" * 50)
        print("กำลังโหลดไฟล์:", filename)
        print("=" * 50)

        add_document(filename)


# =========================
# MAIN
# =========================
def main():
    print()
    print("=" * 50)
    print("Python RAG + Ollama")
    print("=" * 50)

    load_documents()

    print()
    print("=" * 50)
    print("RAG พร้อมใช้งาน")
    print("พิมพ์ exit เพื่อออก")
    print("=" * 50)

    while True:
        question = input("\nถาม: ")

        if question.lower() == "exit":
            print("จบการทำงาน")
            break

        if question.strip() == "":
            continue

        try:
            answer = ask_ollama(question)

            print()
            print("คำตอบ:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

        except Exception as e:
            print()
            print("เกิดข้อผิดพลาด:")
            print(e)


if __name__ == "__main__":
    main()
