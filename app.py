import streamlit as st
from pypdf import PdfReader
import re
import os
import random
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  PDF PARSING (Ayrıştırma) FONKSİYONU - Satır Bazlı Yaklaşım
# ------------------------------------------------------------------
def parse_aws_questions(pdf_content):
    """
    PDF'i satır satır okuyarak soruları ayrıştırır.
    Bu yaklaşım, regex tabanlı yaklaşımdan çok daha dayanıklıdır
    çünkü PDF'teki format tutarsızlıklarını tolere eder.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_content)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return []

    # Ön temizlik: sayfa başlıklarını ve shapingpixel.com linklerini kaldır
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Sayfa başlıklarını atla
        if re.match(r'^=+\s*Page\s+\d+', stripped):
            continue
        if re.match(r'^=+$', stripped):
            continue
        # shapingpixel.com satırlarını atla
        if 'shapingpixel.com' in stripped.lower():
            continue
        # [text layer] gibi işaretleri atla
        if stripped.startswith('[text layer]'):
            continue
        cleaned_lines.append(line)

    questions = []
    current_question = None
    current_options = []
    current_correct = None
    state = "seeking_question"  # seeking_question, reading_question, reading_options

    # Soru başlangıcını yakalayan pattern: "1) " veya "1). " veya "1)Question"
    question_start_pattern = re.compile(r'^(\d+)\s*[\)\.]\s*(.*)')
    # Şık pattern'i: "A. ", "A) ", "A-" ile başlayan (noktalı veya noktasız)
    option_pattern = re.compile(r'^([A-Z])\s*[\)\.\-]?\s+(.+)')
    # Doğru cevap pattern'leri
    correct_answer_patterns = [
        re.compile(r'^Correct\s+Answer\s*:\s*([A-Z]+)', re.IGNORECASE),
        re.compile(r'^Answer\s*\(s\)\s*:\s*([A-Z]+)', re.IGNORECASE),
        re.compile(r'^Answer\s*:\s*([A-Z]+)', re.IGNORECASE),
    ]

    def finalize_question():
        """Toplanan veriyi soru listesine ekler."""
        nonlocal current_question, current_options, current_correct
        if current_question and len(current_options) >= 2:
            # Doğru cevabın tam metnini bul
            correct_full = current_correct if current_correct else ""
            if current_correct and len(current_correct) >= 1:
                first_letter = current_correct[0]
                for opt in current_options:
                    if opt.strip().startswith(first_letter + "."):
                        correct_full = opt
                        break
            
            q_data = {
                'soru': current_question,
                'siklar': current_options.copy(),
                'dogru_cevap': correct_full
            }
            questions.append(q_data)
        
        current_question = None
        current_options = []
        current_correct = None

    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Doğru cevap satırı mı?
        matched_correct = None
        for pattern in correct_answer_patterns:
            m = pattern.match(stripped)
            if m:
                matched_correct = m.group(1).strip()
                break

        if matched_correct:
            current_correct = matched_correct
            i += 1
            continue

        # Yeni soru başlangıcı mı?
        q_match = question_start_pattern.match(stripped)
        if q_match:
            q_num = q_match.group(1)
            q_text = q_match.group(2).strip()

            # Eğer elimizde bekleyen bir soru varsa, onu kaydet
            if current_question is not None:
                finalize_question()

            # Yeni soruyu başlat
            current_question = f"{q_num}) {q_text}" if q_text else f"{q_num})"
            current_options = []
            current_correct = None
            state = "reading_question"
            i += 1
            continue

        # Şık satırı mı? (sadece bir soru topluyorsak)
        if current_question is not None:
            o_match = option_pattern.match(stripped)
            if o_match:
                opt_letter = o_match.group(1)
                opt_text = o_match.group(2).strip()
                # "Most Voted" gibi ekstra metinleri temizle
                opt_text = re.sub(r'Most\s+Voted', '', opt_text, flags=re.IGNORECASE).strip()
                # Çok satırlı şıkları birleştir (sonraki satır küçük harfle devam ediyorsa)
                while i + 1 < len(cleaned_lines):
                    next_line = cleaned_lines[i + 1].strip()
                    if (next_line and 
                        not question_start_pattern.match(next_line) and
                        not option_pattern.match(next_line) and
                        not any(p.match(next_line) for p in correct_answer_patterns) and
                        not next_line.startswith(('Explanation:', 'Correct Answer', 'Answer'))):
                        opt_text += " " + next_line
                        i += 1
                    else:
                        break
                
                current_options.append(f"{opt_letter}. {opt_text}")
                i += 1
                continue

            # Soru metninin devamı olabilir (şık başlamadan önce)
            if len(current_options) == 0 and state == "reading_question":
                # Açıklama, Correct Answer gibi anahtar kelimeleri atla
                if not stripped.startswith(('Explanation:', 'Correct Answer', 'Answer')):
                    # Soru metnine ekle
                    current_question += " " + stripped
        
        i += 1

    # Döngü bittiğinde son soruyu da kaydet
    if current_question is not None:
        finalize_question()

    return questions


# ------------------------------------------------------------------
#  Veritabanı ve Model Fonksiyonları
# ------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    print("Embedding modeli yükleniyor...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding modeli yüklendi.")
    return model

@st.cache_resource
def get_vector_db():
    print("Vektör veritabanı başlatılıyor...")
    client = chromadb.EphemeralClient() 
    print("Vektör veritabanı başlatıldı.")
    return client

def setup_database(client, model, questions_list):
    try:
        collection = client.get_or_create_collection(name="aws_questions")
    except Exception as e:
        st.error(f"Vektör DB koleksiyonu oluşturulamadı: {e}")
        return None

    if collection.count() != len(questions_list):
        print(f"Veritabanı {collection.count()} / {len(questions_list)} soru içeriyor. Yeniden indeksleniyor...")
        if collection.count() > 0:
            client.delete_collection(name="aws_questions")
            collection = client.get_or_create_collection(name="aws_questions")
        
        documents_to_embed = []
        metadatas_for_db = []
        ids_for_db = []
        
        for i, q in enumerate(questions_list):
            content = f"Question: {q['soru']}"
            documents_to_embed.append(content)
            metadatas_for_db.append({"original_index": i})
            ids_for_db.append(f"q_{i}")

        # Batch halinde ekle (ChromaDB limitleri için)
        batch_size = 500
        for start in range(0, len(documents_to_embed), batch_size):
            end = start + batch_size
            embeddings = model.encode(documents_to_embed[start:end])
            collection.add(
                embeddings=embeddings.tolist(),
                documents=documents_to_embed[start:end],
                metadatas=metadatas_for_db[start:end],
                ids=ids_for_db[start:end]
            )
        print("İndeksleme tamamlandı.")
    else:
        print("Veritabanı zaten güncel. İndeksleme atlanıyor.")
        
    return collection

@st.cache_data
def load_and_parse_questions(pdf_path):
    print("PDF ayrıştırılıyor...")
    if not os.path.exists(pdf_path):
        st.error(f"Hata: '{pdf_path}' yolunda PDF bulunamadı.")
        return None
    try:
        with open(pdf_path, "rb") as f:
            return parse_aws_questions(f)
    except Exception as e:
        st.error(f"PDF okuma hatası: {e}")
        return None

# ------------------------------------------------------------------
#  Streamlit Arayüzü
# ------------------------------------------------------------------

st.title("AWS SAA Quiz Bot 🧠☁️")

# --- 1. Yükleme ve Kurulum ---
model = get_embedding_model()
client = get_vector_db()
questions_list = load_and_parse_questions("data/saa_exam.pdf")

if not questions_list:
    st.error("PDF'ten hiç soru okunamadığı için uygulama durduruldu.")
    st.stop()

st.success(f"✅ Başarıyla {len(questions_list)} soru ayrıştırıldı.")

collection = setup_database(client, model, questions_list)
if not collection:
    st.error("Vektör veritabanı kurulamadığı için uygulama durduruldu.")
    st.stop()

# --- Oturum Durumu (Session State) ---
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.questions_to_ask = []
    st.session_state.num_to_ask = 0
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.user_answers = {}

# Stage 1: Konu Seçme VEYA Rastgele
if not st.session_state.quiz_started:
    
    st.info(f"{collection.count()} adet AWS sorusu indekslendi.")
    
    user_topic = st.text_input(
        "Which topic do you want to be quizzed on? (Leave blank for random)", 
        placeholder="e.g., S3 and storage"
    )
    
    num_to_ask_input = st.number_input(
        "How many questions?",
        min_value=1,
        max_value=50,
        value=3
    )
    num_to_ask = int(num_to_ask_input)

    if st.button("Start Quiz"):
        
        retrieved_questions = []
        
        if user_topic.strip():
            with st.spinner(f"Finding the {num_to_ask} best questions about '{user_topic}'..."):
                query_embedding = model.encode([user_topic])[0].tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=num_to_ask
                )
                for metadata in results['metadatas'][0]:
                    idx = metadata['original_index']
                    retrieved_questions.append(questions_list[idx])
        
        else:
            with st.spinner(f"Selecting {num_to_ask} random questions..."):
                num_available = len(questions_list)
                actual_num_to_get = min(num_to_ask, num_available)
                if actual_num_to_get > 0:
                    retrieved_questions = random.sample(questions_list, actual_num_to_get)
        
        if not retrieved_questions:
            st.warning("No questions found. Please check your topic or if the PDF was parsed correctly.")
        else:
            st.session_state.questions_to_ask = retrieved_questions
            st.session_state.num_to_ask = len(retrieved_questions)
            st.session_state.quiz_started = True
            st.session_state.current_question_index = 0
            st.session_state.score = 0
            st.session_state.user_answers = {}
            st.rerun()

# Stage 2: Quiz'i Gösterme
elif st.session_state.quiz_started and st.session_state.current_question_index < st.session_state.num_to_ask:
    
    idx = st.session_state.current_question_index
    q = st.session_state.questions_to_ask[idx] 
    
    st.subheader(f"Question {idx + 1} / {st.session_state.num_to_ask}")
    st.write(q.get('soru', 'Question text not found'))
    
    with st.form(key=f"form_q_{idx}"):
        user_answer = st.radio(
            "Select your answer:",
            q.get('siklar', []),
            key=f"radio_q_{idx}",
            index=None
        )
        submit_button = st.form_submit_button("Submit Answer")

    if submit_button:
        if user_answer is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.user_answers[idx] = user_answer
            
            correct_answer_text = q.get('dogru_cevap', 'Z').strip()
            user_answer_prefix = user_answer.strip()[0]
            
            if len(correct_answer_text) <= 5: 
                 correct_answer_prefix = correct_answer_text
            else:
                 correct_answer_prefix = correct_answer_text[0]

            if user_answer_prefix in correct_answer_prefix:
                st.success("Correct! 🎉")
                st.session_state.score += 1
            else:
                st.error(f"Incorrect. The correct answer was: {q.get('dogru_cevap', 'N/A')}")
            
            st.session_state.current_question_index += 1
            
            if st.session_state.current_question_index < st.session_state.num_to_ask:
                st.button("Next Question")
            else:
                st.button("View Results")

# Stage 3: Sonuç Ekranı
elif st.session_state.quiz_started and st.session_state.current_question_index >= st.session_state.num_to_ask:
    st.balloons()
    st.header("Quiz Finished!")
    st.write(f"You answered {st.session_state.score} out of {st.session_state.num_to_ask} questions correctly.")
    
    if st.button("Start Over"):
        st.session_state.quiz_started = False
        st.session_state.questions_to_ask = []
        st.session_state.num_to_ask = 0
        st.session_state.current_question_index = 0
        st.session_state.score = 0
        st.session_state.user_answers = {}
        st.rerun()