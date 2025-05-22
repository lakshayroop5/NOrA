# -*- coding: utf-8 -*-
"""
NOrA: AI-Powered Lung Cancer Recovery Advisor
Streamlit implementation focused on lung cancer survivorship
"""

import time
import requests
import streamlit as st
from typing import List, Dict, Union
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from langdetect import detect

# Set page configuration
st.set_page_config(
    page_title="NOrA: Lung Cancer Recovery Advisor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🫁 NOrA: Lung Cancer Recovery Advisor")
st.markdown("### *AI-Powered support for lung cancer survivors*")
st.markdown("""
This application provides evidence-based guidance for lung cancer survivors using:
- Medical guidelines from trusted sources
- Latest clinical trials data specific to lung cancer recovery
- Multilingual support (English/French/Arabic)
""")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured!")
    else:
        st.warning("Please enter your Google API Key to continue")
    
    st.markdown("---")
    st.markdown("### About NOrA")
    st.markdown("""
    This tool helps lung cancer survivors navigate post-treatment challenges with:
    - Nutrition advice for lung health
    - Exercise recommendations after lung surgery
    - Managing respiratory symptoms
    - Emotional support specific to lung cancer recovery
    """)

# Integrated lung cancer resources
def get_integrated_resources() -> List[Dict]:
    """
    Provides pre-defined lung cancer recovery resources
    """
    resources = [
        {
            "text": """LUNG CANCER NUTRITION GUIDELINES
Proper nutrition is essential for lung cancer recovery. Patients should focus on protein-rich foods to rebuild tissue after surgery or treatment. Recommended foods include lean meats, fish, eggs, dairy, legumes, and plant proteins. Patients should eat small, frequent meals if experiencing appetite loss or early satiety. Staying hydrated is crucial, aiming for 8-10 cups of fluid daily. Patients should limit processed foods high in sodium and sugar. Supplements should only be taken under medical supervision. Weight maintenance is important - unintentional weight loss can delay recovery and reduce treatment tolerance.""",
            "source": "Lung Cancer Nutrition Guide",
            "page": 1,
            "type": "patient_guide"
        },
        {
            "text": """EXERCISE AFTER LUNG CANCER SURGERY
Gradual return to physical activity is recommended after lung surgery. Start with short walks and gradually increase duration and intensity. Breathing exercises are essential for lung expansion and preventing complications. Pulmonary rehabilitation programs can significantly improve lung function and exercise capacity. Resistance training helps rebuild muscle mass lost during treatment. Exercise should be modified based on individual limitations and treatment side effects. Always consult healthcare providers before starting any exercise program. Signs to stop exercise include severe shortness of breath, chest pain, dizziness, or unusual fatigue.""",
            "source": "Post-Surgical Recovery Guidelines",
            "page": 1,
            "type": "patient_guide"
        },
        {
            "text": """MANAGING RESPIRATORY SYMPTOMS
Breathlessness is common after lung cancer treatment. Techniques to manage include pursed-lip breathing, diaphragmatic breathing, and paced breathing during activity. Position changes can help ease breathing difficulties - try leaning forward with arms supported. Energy conservation techniques can reduce breathlessness during daily activities. Supplemental oxygen may be prescribed for some patients. Persistent cough can be managed with hydration, humidification, and prescribed medications. Report any changes in sputum color or blood in sputum immediately. Pulmonary function tests should be performed regularly to monitor lung capacity.""",
            "source": "Respiratory Management Protocol",
            "page": 1,
            "type": "patient_guide"
        },
        {
            "text": """EMOTIONAL SUPPORT FOR LUNG CANCER SURVIVORS
Psychological distress is common among lung cancer survivors. Symptoms may include anxiety, depression, fear of recurrence, and adjustment difficulties. Support groups specifically for lung cancer patients can provide valuable emotional support and practical advice. Individual counseling or therapy may help process cancer-related trauma and develop coping strategies. Mind-body practices like meditation, yoga, and relaxation techniques can reduce stress and improve quality of life. Family members should be included in psychological support planning. Screening for distress should be part of routine follow-up care. Addressing stigma associated with lung cancer is an important aspect of emotional recovery.""",
            "source": "Psychosocial Oncology Handbook",
            "page": 1,
            "type": "patient_guide"
        },
        {
            "text": """LUNG CANCER SURVEILLANCE RECOMMENDATIONS
Regular follow-up is essential for early detection of recurrence or complications. Follow-up schedule typically includes visits every 3-6 months for the first 2 years, then every 6-12 months for years 3-5. CT scans are usually recommended every 6-12 months for the first 2 years, then annually. Patients should report new or worsening symptoms promptly, including persistent cough, chest pain, shortness of breath, unexplained weight loss, or bone pain. Smoking cessation support should continue throughout survivorship. Secondary cancer screening should follow age-appropriate guidelines. Pulmonary function tests may be performed periodically, especially if additional treatment is considered.""",
            "source": "Surveillance Protocol for Lung Cancer Survivors",
            "page": 1,
            "type": "patient_guide"
        },
        {
            "text": """MANAGING TREATMENT SIDE EFFECTS
Fatigue is one of the most common side effects and may persist for months after treatment. Energy conservation strategies include prioritizing activities, scheduling rest periods, and delegating tasks. Peripheral neuropathy may occur with certain chemotherapies, requiring safety precautions and possibly medication for symptom management. Radiation pneumonitis can develop weeks to months after chest radiation, presenting as cough, fever, or shortness of breath requiring prompt medical attention. Immune-related adverse events can occur with immunotherapy and require immediate reporting of new symptoms. Hormonal changes may affect both men and women, potentially causing mood changes, sleep disturbances, and other symptoms.""",
            "source": "Treatment Side Effects Management Guide",
            "page": 1,
            "type": "patient_guide"
        }
    ]
    return resources

# Function to fetch clinical trials data
def get_clinical_trials(search_terms: List[str], max_results: int = 10) -> List[Dict]:
    """Fetch lung cancer specific clinical trials data"""
    all_studies = []
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, term in enumerate(search_terms):
        try:
            status_text.text(f"Fetching trials for: {term}")
            params = {
                "query.term": f"{term} AND AREA[StudyType]Interventional",
                "pageSize": max_results,
                "format": "json"
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            studies = response.json().get("studies", [])
            st.write(f"Found {len(studies)} trials for '{term}'")

            for study in studies:
                protocol = study.get("protocolSection", {})
                all_studies.append({
                    "text": f"STUDY: {protocol.get('identificationModule',{}).get('briefTitle','')}\n"
                            f"DESCRIPTION: {protocol.get('descriptionModule',{}).get('detailedDescription','')}",
                    "source": "ClinicalTrials.gov",
                    "nctId": protocol.get("identificationModule", {}).get("nctId"),
                    "type": "clinical_trial"
                })
            
            # Update progress
            progress_bar.progress((i + 1) / len(search_terms))
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            st.error(f"API Error for '{term}': {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    return all_studies

# Function to split text into chunks
def chunk_documents(documents: List[Dict], chunk_size=1000, overlap=200) -> List[Dict]:
    """
    Splits each document into chunks of around `chunk_size` words,
    with `overlap` words between chunks to preserve context.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            metadata = doc.copy()
            metadata.pop("text")  # remove full text to avoid redundancy
            metadata["chunk_id"] = f"{metadata.get('source', 'doc')}_{i}"

            chunks.append({
                "text": chunk,
                "metadata": metadata
            })

    return chunks

# Function to get embeddings
def get_embeddings(texts: Union[str, List[str]], task_type="RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Calls Gemini embedding API to convert text into vector(s).
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        embeddings.append(result['embedding'])

    return embeddings

# Function to generate response
def generate_response(query: str, language: str = None, collection=None) -> str:
    """Enhanced RAG response generator for lung cancer survivors"""
    # Auto-detect language
    if not language:
        try:
            language = detect(query)
            language = language if language in ['en', 'fr', 'ar'] else 'en'
        except:
            language = 'en'

    # Retrieve context
    query_embedding = get_embeddings(query, "RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # Build context
    context = "LUNG CANCER RECOVERY KNOWLEDGE BASE:\n"
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context += f"\n🔍 Source: {meta['source']} (Relevance: {1 - results['distances'][0][i]:.2f})\n"
        context += f"{doc}\n{'='*50}\n"

    # Language-specific prompts with lung cancer focus
    prompts = {
        'en': """You are a specialized lung cancer recovery advisor. Based STRICTLY on the provided sources:
{context}

Question: {question}

Provide a focused response addressing ONLY the specific query. Format your response:
1. Direct answer to the question (2-3 sentences)
2. Evidence-based recommendations from sources (bullet points)
3. Brief source citations [in brackets]
4. ONE clear statement about when to consult a healthcare provider

Important:
- Stay focused on the specific question
- Only use information from provided sources
- Keep responses concise and practical
- Avoid general medical advice not found in sources""",

        'fr': """Vous êtes un conseiller spécialisé dans le rétablissement du cancer du poumon. En vous basant STRICTEMENT sur les sources fournies :
{context}

Question : {question}

Fournissez une réponse ciblée qui aborde UNIQUEMENT la question spécifique. Format de réponse :
1. Réponse directe à la question (2-3 phrases)
2. Recommandations basées sur les sources (points)
3. Brèves citations des sources [entre crochets]
4. UNE déclaration claire sur quand consulter un professionnel de santé

Important :
- Restez concentré sur la question spécifique
- Utilisez uniquement les informations des sources fournies
- Gardez les réponses concises et pratiques
- Évitez les conseils médicaux généraux non présents dans les sources""",

        'ar': """أنت مستشار متخصص في التعافي من سرطان الرئة. استناداً بشكل صارم إلى المصادر المقدمة:
{context}

السؤال: {question}

قدم إجابة مركزة تتناول السؤال المحدد فقط. تنسيق الإجابة:
1. إجابة مباشرة على السؤال (2-3 جمل)
2. توصيات مستندة إلى المصادر (نقاط)
3. اقتباسات موجزة من المصادر [بين قوسين]
4. عبارة واحدة واضحة حول متى يجب استشارة مقدم الرعاية الصحية

مهم:
- ركز على السؤال المحدد
- استخدم فقط المعلومات من المصادر المقدمة
- احتفظ بالإجابات موجزة وعملية
- تجنب النصائح الطبية العامة غير الموجودة في المصادر"""
    }

    # Generate response
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompts[language].format(context=context, question=query),
        generation_config={
            "temperature": 0.3,  # Reduced for more focused responses
            "top_p": 0.8,      # Slightly reduced for better coherence
            "max_output_tokens": 1000  # Reduced to encourage conciseness
        },
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    )

    return response.text

# Function to translate text
def translate_text(text: str, target_lang: str) -> str:
    """Use Gemini for translation"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"Translate this accurately to {target_lang} for medical professionals specializing in lung cancer:\n{text}",
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    )
    return response.text

# Function to generate Arabic response
def generate_arabic_response(query: str, collection=None):
    # Step 1: Translate query to English
    translated_query = translate_text(query, "en")

    # Step 2: Get English response
    english_response = generate_response(translated_query, "en", collection)

    # Step 3: Translate back to Arabic
    arabic_response = translate_text(english_response, "ar")

    # Step 4: Format for RTL
    return f"<div style='direction: rtl; text-align: right; font-family: Tahoma;'>⚠️ ملاحظة: تمت الترجمة الآلية<br><br>{arabic_response}</div>"

# Main application flow
def main():
    # Check if API key is provided
    if not api_key:
        st.warning("Please enter your Google API Key in the sidebar to continue")
        return
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    
    # Data loading section
    st.header("📚 Data Sources")
    
    # Lung cancer specific search terms
    lung_cancer_terms = [
        "lung cancer survivorship",
        "post lung surgery recovery",
        "lung cancer nutrition",
        "respiratory rehabilitation after lung cancer",
        "lung cancer fatigue management",
        "breathing exercises after lung cancer",
        "lung cancer emotional support",
        "lung cancer recurrence prevention"
    ]
    
    # Show resource status
    if st.session_state.collection:
        # Display success message
        st.success("✅ Knowledge base is loaded and ready to use")
        
        # Add a disabled button to indicate loaded state
        st.button("Knowledge Base Loaded", disabled=True)
        
        # Add a small reload button if needed
        if st.button("🔄 Reload Knowledge Base"):
            st.session_state.collection = None
            st.rerun()
    else:
        # Process button
        if st.button("Load Knowledge Base"):
            with st.spinner("Processing data..."):
                # Get integrated resources
                integrated_resources = get_integrated_resources()
                st.success(f"Loaded {len(integrated_resources)} integrated lung cancer resources")
                
                # Fetch clinical trials data
                with st.expander("Clinical Trials Data"):
                    clinical_data = get_clinical_trials(lung_cancer_terms)
                    st.success(f"Retrieved {len(clinical_data)} lung cancer clinical trials")
                
                # Combine and chunk data
                all_docs = integrated_resources + clinical_data
                processed_chunks = chunk_documents(all_docs)
                st.success(f"Created {len(processed_chunks)} text chunks from all documents")
                
                # Store in ChromaDB
                with st.spinner("Creating vector database..."):
                    # Initialize persistent ChromaDB storage
                    client = chromadb.PersistentClient(path="./chroma_db")
                    
                    # Create or load collection
                    collection = client.get_or_create_collection(
                        name="lung_cancer_recovery_resources",
                        metadata={"hnsw:space": "cosine"}
                    )
                    
                    # Store chunks in batches
                    batch_size = 10
                    progress_bar = st.progress(0)
                    
                    for i in range(0, len(processed_chunks), batch_size):
                        batch = processed_chunks[i:i + batch_size]
                        
                        batch_embeddings = get_embeddings(
                            [chunk["text"] for chunk in batch],
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        
                        metadatas = [chunk["metadata"] for chunk in batch]
                        ids = [f"doc_{i+j}" for j in range(len(batch))]
                        
                        collection.add(
                            documents=[chunk["text"] for chunk in batch],
                            metadatas=metadatas,
                            embeddings=batch_embeddings,
                            ids=ids
                        )
                        
                        # Update progress
                        progress_bar.progress(min(1.0, (i + batch_size) / len(processed_chunks)))
                    
                    st.session_state.collection = collection
                    st.success(f"Stored {collection.count()} chunks in vector database")
                    
                    # Force a rerun to update the UI and show the disabled button
                    st.rerun()
    
    # Chat interface
    st.header("💬 Lung Cancer Recovery Assistant")
    
    # Language selection
    language = st.selectbox(
        "Select Model's Language",
        options=["en", "fr", "ar"],
        format_func=lambda x: {"en": "English", "fr": "French", "ar": "Arabic"}[x]
    )
    
    # Show sources option
    show_sources = st.checkbox("Show reference sources", value=True)
    
    # Add refresh button for chat
    if st.button("🔄 Refresh Chat"):
        st.session_state.chat_history = []
        # Don't try to modify user_input here
        st.rerun()
    
    # Create a container for chat history
    chat_container = st.container()
    
    # Create a container for the input box at the bottom
    input_container = st.container()
    
    # Display chat history in the chat container
    with chat_container:
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            if language == "ar":
                st.markdown(chat['bot'], unsafe_allow_html=True)
            else:
                st.markdown(f"**NOrA:** {chat['bot']}")
    
    # Place the input box at the bottom
    with input_container:
        # Initialize callback to handle form submission
        def handle_input():
            if st.session_state.user_input and st.session_state.collection:
                # Store the query before processing
                query = st.session_state.user_input
                
                # Reset the input by setting the session state BEFORE the widget is rendered again
                st.session_state.user_input = ""
                
                # Add to chat history immediately to show the user's query
                st.session_state.chat_history.append({"user": query, "bot": None})
                
                # Force a rerun to show the user's message before processing
                st.rerun()

        # Create the input box with a callback
        user_input = st.text_input(
            "Ask a question about lung cancer recovery:", 
            key="user_input",
            on_change=handle_input
        )
        
        # If collection is not initialized
        if not st.session_state.collection and user_input:
            st.warning("Please load the knowledge base first before asking questions")
    
    # Process any pending response (this runs after the UI is rendered)
    # This ensures the loading indicator appears in the right place in the chat
    if st.session_state.chat_history and st.session_state.chat_history[-1]["bot"] is None:
        # Get the last query that needs processing
        last_query = st.session_state.chat_history[-1]["user"]
        
        # Create a placeholder for the loading animation in the chat container
        with chat_container:
            with st.status("Generating response...", expanded=True) as status:
                st.write("🧠 Analyzing your question...")
                time.sleep(0.5)  # Small delay for better UX
                
                # Generate response based on language
                st.write("🔍 Searching knowledge base...")
                if language == "ar":
                    response = generate_arabic_response(last_query, st.session_state.collection)
                else:
                    response = generate_response(last_query, language, st.session_state.collection)
                
                st.write("✏️ Formatting answer...")
                # Filter sources if needed
                if not show_sources:
                    response = "\n".join([
                        line for line in response.split("\n")
                        if not line.lower().startswith("source:")
                    ])
                
                # Update the chat history with the response
                st.session_state.chat_history[-1]["bot"] = response
                status.update(label="Response ready!", state="complete", expanded=False)
                
                # Force a rerun to show the complete chat history
                st.rerun()

if __name__ == "__main__":
    main()
