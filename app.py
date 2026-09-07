import streamlit as st
from pypdf import PdfReader
import re
import os
import random
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  PDF PARSING FUNCTION (No Explanations Version)
# ------------------------------------------------------------------
def parse_aws_questions(pdf_content):
    """
    Reads PDF file content, combines text, and returns a list of questions
    using regex parsing without explanations.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_content)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text += "\n"
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return []

    questions = []
    
    # Regex pattern preserved for correct PDF structure capture
    pattern = re.compile(
        r"^(\d+)\s*\)\s(.*?)" +         # Group 1 (num), optional space, ), space, Group 2 (Q Text)
        r"(?=\n[A-Z]\.)" +              # Anchor: Start of options (\nA.)
        r"((?:(?!\nCorrect Answer:|\nExplanation:).)*)" + # Group 3: Options
        r"(\nExplanation:((?:(?!\nCorrect Answer:).)*))?" + # Group 4/5: Explanation (Before)
        r"(\nCorrect Answer:\s*([A-Z]{1,5}))" +  # Group 6/7: Answer (Required, supports multi-answer like AD/BC)
        r"((?:(?!\n\d+\)|$).)*)?" +     # Group 8: Explanation (After)
        r"(?=\n\d+\s*\)|$)",            # End Anchor
        re.DOTALL | re.IGNORECASE
    )
    
    chunks = re.split(r'\n(?=\d+\s*\))', text)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        match = pattern.search(chunk)
        
        if match:
            (q_num, q_text, options_block,
             _full_expl_before, _expl_text_before,
             _full_answer_line, correct_letter,
             _expl_text_after) = match.groups()
            
            # Combine question number and text
            question = f"{q_num}) {q_text.strip()}"
            
            cleaned_options = []
            option_pattern = re.compile(r"([A-Z])\.(.*?)(?=\n[A-Z]\.|$)", re.DOTALL | re.IGNORECASE)
            
            for opt_match in option_pattern.finditer(options_block.strip()):
                opt_letter, opt_text = opt_match.groups()
                opt_cleaned = re.sub(r'Most Voted', '', opt_text, flags=re.IGNORECASE).strip()
                opt_final = re.sub(r'\s*\n\s*', ' ', opt_cleaned).strip()
                cleaned_options.append(f"{opt_letter}. {opt_final}")

            correct_answer_full_text = ""
            correct_letter = correct_letter.strip()
            
            # Find the full text of the correct answer
            first_correct_letter = correct_letter[0]
            for opt in cleaned_options:
                if opt.strip().startswith(first_correct_letter + "."):
                    correct_answer_full_text = opt
                    break
            
            if not correct_answer_full_text:
                correct_answer_full_text = correct_letter

            # Explanation key completely removed from data structure
            q_data = {
                'question': question,
                'options': cleaned_options,
                'correct_answer': correct_answer_full_text
            }
            questions.append(q_data)
        
        else:
            pass # Silently skip failures

    if not questions:
        st.error("No questions could be parsed. Please check the PDF format or Regex pattern.")

    return questions

# ------------------------------------------------------------------
#  Database and Model Functions
# ------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
    return model

@st.cache_resource
def get_vector_db():
    print("Initializing vector database...")
    client = chromadb.EphemeralClient() 
    print("Vector database initialized.")
    return client

def setup_database(client, model, questions_list):
    try:
        collection = client.get_or_create_collection(name="aws_questions")
    except Exception as e:
        st.error(f"Could not create vector DB collection: {e}")
        return None

    if collection.count() != len(questions_list):
        print(f"Database contains {collection.count()} / {len(questions_list)} questions. Re-indexing...")
        if collection.count() > 0:
            client.delete_collection(name="aws_questions")
            collection = client.get_or_create_collection(name="aws_questions")
        
        documents_to_embed = []
        metadatas_for_db = []
        ids_for_db = []
        
        for i, q in enumerate(questions_list):
            # Indexing only on Question text (Explanations removed)
            content = f"Question: {q['question']}"
            documents_to_embed.append(content)
            metadatas_for_db.append({"original_index": i})
            ids_for_db.append(f"q_{i}")

        embeddings = model.encode(documents_to_embed)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents_to_embed,
            metadatas=metadatas_for_db,
            ids=ids_for_db
        )
        print("Indexing complete.")
    else:
        print("Database is already up to date. Skipping indexing.")
        
    return collection

@st.cache_data
def load_and_parse_questions(pdf_path):
    print("Parsing PDF...")
    if not os.path.exists(pdf_path):
        st.error(f"Error: PDF not found at '{pdf_path}'.")
        return None
    try:
        with open(pdf_path, "rb") as f:
            return parse_aws_questions(f)
    except Exception as e:
        st.error(f"PDF reading error: {e}")
        return None

# ------------------------------------------------------------------
#  Streamlit Interface
# ------------------------------------------------------------------

st.title("AWS SAA Quiz Bot 🧠☁️")

# --- 1. Loading and Setup ---
model = get_embedding_model()
client = get_vector_db()
questions_list = load_and_parse_questions("data/saa_exam.pdf")

if not questions_list:
    st.error("No questions could be read from PDF. Application stopped.")
    st.stop()

collection = setup_database(client, model, questions_list)
if not collection:
    st.error("Vector database could not be set up. Application stopped.")
    st.stop()

# --- Session State ---
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.questions_to_ask = []
    st.session_state.num_to_ask = 0
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.user_answers = {}

# Stage 1: Topic Selection OR Random
if not st.session_state.quiz_started:
    
    st.info(f"{collection.count()} AWS questions indexed.")
    
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

# Stage 2: Show Quiz
elif st.session_state.quiz_started and st.session_state.current_question_index < st.session_state.num_to_ask:
    
    idx = st.session_state.current_question_index
    q = st.session_state.questions_to_ask[idx] 
    
    st.subheader(f"Question {idx + 1} / {st.session_state.num_to_ask}")
    st.write(q.get('question', 'Question text not found'))
    
    with st.form(key=f"form_q_{idx}"):
        user_answer = st.radio(
            "Select your answer:",
            q.get('options', []),
            key=f"radio_q_{idx}",
            index=None
        )
        submit_button = st.form_submit_button("Submit Answer")

    if submit_button:
        if user_answer is None:
            st.warning("Please select an answer.")
        else:
            st.session_state.user_answers[idx] = user_answer
            
            correct_answer_text = q.get('correct_answer', 'Z').strip()
            user_answer_prefix = user_answer.strip()[0]
            
            # If answer is just a letter like 'A' or 'AD'
            if len(correct_answer_text) <= 5: 
                 correct_answer_prefix = correct_answer_text
            # If answer is full text like 'A. ...'
            else:
                 correct_answer_prefix = correct_answer_text[0]

            # Check if user's answer first letter is in the correct answer letters
            if user_answer_prefix in correct_answer_prefix:
                st.success("Correct! 🎉")
                st.session_state.score += 1
            else:
                st.error(f"Incorrect. The correct answer was: {q.get('correct_answer', 'N/A')}")
            
            # NOTE: st.info(Explanation) block has been completely removed from here.
            
            st.session_state.current_question_index += 1
            
            if st.session_state.current_question_index < st.session_state.num_to_ask:
                st.button("Next Question")
            else:
                st.button("View Results")

# Stage 3: Results Screen
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