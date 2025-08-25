import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
import io

# -------------------- Config --------------------
API_KEY = "AIzaSyAd-pJNXnxFNCr-zrb967WAILMuD6O-QxQ"
st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# -------------------- Helper Functions --------------------
def extract_paragraphs(uploaded_file):
    """
    FIXED: Now takes the uploaded_file object directly, not just the name
    """
    paragraphs = []
    
    # Get file extension from the name
    file_extension = uploaded_file.name.lower().split('.')[-1]

    if file_extension == "txt":
        # Read bytes and decode
        text = uploaded_file.read().decode("utf-8")
        paragraphs = text.split("\n\n")

    elif file_extension == "pdf":
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        paragraphs = text.split("\n\n")

    elif file_extension == "docx":
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Assign IDs
    indexed_paragraphs = [{"id": idx, "text": para.strip()} for idx, para in enumerate(paragraphs) if para.strip()]
    return indexed_paragraphs


def summarize_file(uploaded_file, model_name="gemini-2.5-flash"):
    """
    FIXED: Now takes uploaded_file object and handles it properly
    """
    # Reset file pointer to beginning (important!)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    
    # Determine MIME type based on file extension
    file_extension = uploaded_file.name.lower().split('.')[-1]
    if file_extension == 'pdf':
        mime_type = 'application/pdf'
    elif file_extension == 'docx':
        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    else:  # txt
        mime_type = 'text/plain'

    file_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type,
    )

    prompt = "Please provide a clear and concise summary of the document in bullet points."

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model=model_name,
        contents=[file_part, prompt]
    )

    # Split response into sentences/points
    summary_points = [p.strip() for p in response.text.split("\n") if p.strip()]
    return summary_points

def map_summary_to_paragraphs(paragraphs, summary_sentences):
    para_texts = [p["text"] for p in paragraphs]
    para_embeddings = model.encode(para_texts, convert_to_tensor=True)
    summary_embeddings = model.encode(summary_sentences, convert_to_tensor=True)

    mapping = []
    for idx, s_emb in enumerate(summary_embeddings):
        cos_scores = util.cos_sim(s_emb, para_embeddings)[0]
        best_match_idx = int(cos_scores.argmax())
        mapping.append({
            "summary_sentence": summary_sentences[idx],
            "reference_paragraph_id": paragraphs[best_match_idx]["id"],
            "reference_text": paragraphs[best_match_idx]["text"]
        })
    return mapping

# -------------------- Streamlit UI --------------------
st.title("📄 Legal Document Summarizer with Paragraph Mapping")

uploaded_file = st.file_uploader("Upload your legal document (.pdf, .docx, .txt)", type=["pdf","docx","txt"])

if uploaded_file:
    with st.spinner("Extracting paragraphs..."):
        st.write(f"Processing: {uploaded_file.name}")
        # FIXED: Pass the file object, not just the name
        paragraphs = extract_paragraphs(uploaded_file)
    
    st.success(f"✅ Extracted {len(paragraphs)} paragraphs.")

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            # FIXED: Pass the file object, not just the name
            summary_points = summarize_file(uploaded_file)
        st.success("✅ Summary generated!")

        # Map summary to paragraphs
        with st.spinner("Mapping summary points to document paragraphs..."):
            mapping = map_summary_to_paragraphs(paragraphs, summary_points)
        st.success("✅ Mapping completed!")

        # -------------------- Display Split Screen --------------------
        st.markdown("---")
        
        # Initialize session state for highlighting
        if "highlighted_paragraph" not in st.session_state:
            st.session_state.highlighted_paragraph = None
        
        # Add custom CSS for better styling and scrolling
        st.markdown("""
        <style>
        .paragraph-box {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #74b9ff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            line-height: 1.6;
            transition: all 0.3s ease;
        }
        
        .paragraph-highlighted {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%) !important;
            color: white !important;
            border-left: 4px solid #fff !important;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(243, 156, 18, 0.5) !important;
            animation: highlight-pulse 1.5s ease-in-out;
        }
        
        @keyframes highlight-pulse {
            0% { box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
            50% { box-shadow: 0 8px 20px rgba(243, 156, 18, 0.6); }
            100% { box-shadow: 0 4px 12px rgba(243, 156, 18, 0.5); }
        }
        
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            margin: 8px 0 !important;
            font-weight: 500 !important;
            text-align: left !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        }
        </style>
        """, unsafe_allow_html=True)
            
        left_col, right_col = st.columns([1.3, 1], gap="large")

        with left_col:
            st.markdown("### 📑 Document Content")
            
            # Create a proper scrollable container using Streamlit's container
            container = st.container(height=500)
            with container:
                for p in paragraphs:
                    # Check if this paragraph should be highlighted
                    highlight_class = "paragraph-highlighted" if st.session_state.highlighted_paragraph == p['id'] else ""
                    
                    st.markdown(f"""
                    <div class="paragraph-box {highlight_class}">
                        {p['text']}
                    </div>
                    """, unsafe_allow_html=True)

        with right_col:
            st.markdown("### 📝 Summary Points")
            
            # Create scrollable summary container
            summary_container = st.container(height=500)
            with summary_container:
                for i, m in enumerate(mapping):
                    # Create unique key for each button
                    button_key = f"summary_btn_{i}_{m['reference_paragraph_id']}"
                    
                    # Use columns to make button behavior more predictable
                    col1, = st.columns([1])
                    with col1:
                        if st.button(
                            f"📌 {m['summary_sentence']}", 
                            key=button_key,
                            help=f"Click to highlight paragraph {m['reference_paragraph_id']}",
                            use_container_width=True
                        ):
                            # Set the highlighted paragraph and force refresh
                            st.session_state.highlighted_paragraph = m['reference_paragraph_id']
                            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
