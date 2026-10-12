<?php
header('Content-Type: application/json; charset=utf-8');
set_time_limit(300);

// =========================
// CONFIG
// =========================
const DB_HOST = 'localhost';
const DB_USER = 'rag_user';
const DB_PASS = '123456';
const DB_NAME = 'rag_db';

const OLLAMA_URL = 'http://127.0.0.1:11434';
const LLM_MODEL = 'qwen2.5:3b'; #ชื่อโมเดลที่ติดตั้งไว้ในollama
const EMBED_MODEL = 'bge-m3';   #ชื่อโมเดลที่ติดตั้งไว้ในollama
const TOP_K = 7;

// =========================
// ฟังก์ชันเรียก Ollama ผ่าน HTTP
// =========================
function ollama($path, $payload)
{
    $ch = curl_init(OLLAMA_URL . $path);
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
        CURLOPT_POSTFIELDS => json_encode($payload, JSON_UNESCAPED_UNICODE),
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 280,
    ]);
    $body = curl_exec($ch);

    if ($body === false) {
        $err = curl_error($ch);
        curl_close($ch);
        throw new Exception('เชื่อมต่อ Ollama ไม่ได้: ' . $err);
    }
    curl_close($ch);

    $data = json_decode($body, true);
    if (isset($data['error'])) {
        throw new Exception($data['error']);
    }
    return $data;
}

function chat($prompt)
{
    $r = ollama('/api/chat', [
        'model' => LLM_MODEL,
        'messages' => [['role' => 'user', 'content' => $prompt]],
        'stream' => false,
        'options' => ['temperature' => 0],
    ]);
    return $r['message']['content'];
}

function embed($text)
{
    $r = ollama('/api/embed', ['model' => EMBED_MODEL, 'input' => $text]);
    return $r['embeddings'][0];
}

// =========================
// ค้นหา chunk ที่ใกล้เคียงจาก MySQL (cosine similarity)
// =========================
function cosine($a, $b)
{
    $dot = 0.0; $na = 0.0; $nb = 0.0;
    $n = min(count($a), count($b));
    for ($i = 0; $i < $n; $i++) {
        $dot += $a[$i] * $b[$i];
        $na += $a[$i] * $a[$i];
        $nb += $b[$i] * $b[$i];
    }
    return ($na == 0 || $nb == 0) ? 0 : $dot / (sqrt($na) * sqrt($nb));
}

function search_rag($question)
{
    $q = embed($question);

    $pdo = new PDO(
        'mysql:host=' . DB_HOST . ';dbname=' . DB_NAME . ';charset=utf8mb4',
        DB_USER,
        DB_PASS,
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
    );

    $rows = $pdo->query(
        'SELECT chunk_text, embedding FROM document_chunks WHERE embedding IS NOT NULL'
    )->fetchAll(PDO::FETCH_ASSOC);

    $scored = [];
    foreach ($rows as $row) {
        $vec = json_decode($row['embedding'], true);
        $scored[] = ['text' => $row['chunk_text'], 'score' => cosine($q, $vec)];
    }

    usort($scored, fn($x, $y) => $y['score'] <=> $x['score']);

    return array_column(array_slice($scored, 0, TOP_K), 'text');
}

// =========================
// MAIN
// =========================
try {
    $input = json_decode(file_get_contents('php://input'), true);
    $question = trim($input['question'] ?? '');
    $mode = $_GET['mode'] ?? 'plain';

    if ($question === '') {
        throw new Exception('กรุณาพิมพ์คำถาม');
    }

    if ($mode === 'rag') {
        $contexts = search_rag($question);

        if (count($contexts) === 0) {
            $out = ['answer' => 'ไม่พบข้อมูลในเอกสาร', 'contexts' => []];
        } else {
            $context_text = implode("\n\n", $contexts);

            $prompt = <<<PROMPT
คุณเป็นผู้ช่วย AI

ให้ตอบโดยใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น
ห้ามใช้ความรู้นอกเอกสาร และห้ามเดา

ถ้า Context ไม่มีข้อมูลที่เกี่ยวข้อง
ให้ตอบว่า:

"ไม่พบข้อมูลในเอกสาร"

Context:

{$context_text}

คำถาม:

{$question}

คำตอบ:
PROMPT;

            $out = ['answer' => chat($prompt), 'contexts' => $contexts];
        }
    } else {
        // โหมดไม่ใช้ RAG: ถามโมเดลตรงๆ
        $out = ['answer' => chat($question)];
    }

    echo json_encode($out, JSON_UNESCAPED_UNICODE);

} catch (Throwable $e) {
    echo json_encode(
        ['answer' => 'เกิดข้อผิดพลาด: ' . $e->getMessage(), 'contexts' => []],
        JSON_UNESCAPED_UNICODE
    );
}
