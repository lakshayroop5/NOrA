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
        'en': """As a lung cancer specialist, provide detailed advice using ONLY these sources:
{context}
Question: {question}
Structure your response:
1. Summary of key findings specific to lung cancer recovery
2. Specific recommendations for lung cancer survivors
3. Source citations
4. When to consult a pulmonologist or oncologist""",

        'fr': """En tant que spécialiste du cancer du poumon, fournissez des conseils détaillés en utilisant UNIQUEMENT ces sources :
{context}
Question : {question}
Structurez votre réponse :
1. Résumé des découvertes spécifiques à la récupération du cancer du poumon
2. Recommandations spécifiques pour les survivants du cancer du poumon
3. Citations des sources
4. Quand consulter un pneumologue ou un oncologue""",

        'ar': """بصفتك أخصائي سرطان الرئة، قدم نصائح مفصلة باستخدام هذه المصادر فقط:
{context}
السؤال: {question}
هيكل الإجابة:
1. ملخص النتائج الخاصة بالتعافي من سرطان الرئة
2. توصيات محددة للناجين من سرطان الرئة
3. استشهادات المصادر
4. متى تستشير طبيب الرئة أو أخصائي الأورام"""
    }

    # Generate response
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(
        prompts[language].format(context=context, question=query),
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 2000
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
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
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
    
    # Chat interface
    st.header("💬 Lung Cancer Recovery Assistant")
    
    # Language selection
    language = st.selectbox(
        "Select language",
        options=["en", "fr", "ar"],
        format_func=lambda x: {"en": "English", "fr": "French", "ar": "Arabic"}[x]
    )
    
    # Show sources option
    show_sources = st.checkbox("Show reference sources", value=True)
    
    # Chat input
    user_input = st.text_input("Ask a question about lung cancer recovery:")
    
    if user_input and st.session_state.collection:
        with st.spinner("Generating response..."):
            # Generate response based on language
            if language == "ar":
                response = generate_arabic_response(user_input, st.session_state.collection)
            else:
                response = generate_response(user_input, language, st.session_state.collection)
            
            # Filter sources if needed
            if not show_sources:
                response = "\n".join([
                    line for line in response.split("\n")
                    if not line.lower().startswith("source:")
                ])
            
            # Add to chat history
            st.session_state.chat_history.append({"user": user_input, "bot": response})
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        if language == "ar":
            st.markdown(chat['bot'], unsafe_allow_html=True)
        else:
            st.markdown(f"**NOrA:** {chat['bot']}")
    
    # If collection is not initialized
    if not st.session_state.collection and user_input:
        st.warning("Please load the knowledge base first before asking questions")

if __name__ == "__main__":
    main()