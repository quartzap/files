import streamlit as st
import PyPDF2
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_embedding(text, model):
    """Generate embeddings for input text"""
    return model.encode(text)


def generate_questions_with_gemini(context, api_key, num_questions=5):
    """Generate interview questions using Google's Gemini API"""
    genai.configure(api_key=api_key)

    # Configure the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }

    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )

    prompt = f"""
    Generate {num_questions} specific and relevant interview questions based on both 
    the job description and the candidate's profile below. The questions should:
    1. Be tailored to assess the candidate's fit for this specific role
    2. Reference specific skills or experiences from the candidate's profile
    3. Probe for examples that demonstrate required competencies
    4. Include technical questions relevant to the position
    5. Avoid generic questions that could be asked to any candidate
    FORMAT: Return only numbered questions (1-{num_questions}), with no additional text.
    CONTEXT:
    {context}
    """

    try:
        response = model.generate_content(prompt)
        questions_text = response.text.strip()

        # Split by newlines and/or numbers to get individual questions
        import re
        questions = re.split(r'\n+|\d+\.', questions_text)
        questions = [q.strip() for q in questions if q.strip()]

        # Ensure we have exactly num_questions
        if len(questions) < num_questions:
            # Add generic questions if not enough were generated
            generic_questions = [
                "Can you tell me more about your experience with this technology?",
                "How would you handle challenges in this role?",
                "What interests you most about this position?",
                "How do your skills align with our requirements?",
                "Can you describe a relevant project you worked on?",
                "What has been your most significant professional achievement?",
                "How do you stay updated with industry trends?",
                "What's your approach to problem-solving?",
                "How do you handle tight deadlines?",
                "Where do you see yourself in five years?"
            ]
            questions.extend(generic_questions[:(num_questions - len(questions))])

        return questions[:num_questions]

    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return [f"Could not generate questions: {str(e)}"]


def create_download_link(pdf_bytes, filename):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'


def generate_pdf(candidate_name, job_desc_text, questions):
    """Generate a PDF with the questions using ReportLab for better Unicode support"""
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Add custom styles - Check if style exists before adding
    if 'Title' not in styles:
        styles.add(ParagraphStyle(name='Title',
                                  fontName='Helvetica-Bold',
                                  fontSize=16,
                                  alignment=TA_CENTER,
                                  spaceAfter=12))

    if 'Subtitle' not in styles:
        styles.add(ParagraphStyle(name='Subtitle',
                                  fontName='Helvetica-Bold',
                                  fontSize=12,
                                  spaceAfter=6))

    if 'Normal' not in styles:
        styles.add(ParagraphStyle(name='Normal',
                                  fontName='Helvetica',
                                  fontSize=10,
                                  spaceAfter=10))

    elements = []

    # Add title
    title = Paragraph(f"Interview Questions for {candidate_name}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Add timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = Paragraph(f"Generated on: {current_time}", styles['Normal'])
    elements.append(timestamp)
    elements.append(Spacer(1, 12))

    # Add job description summary
    elements.append(Paragraph("Job Description Summary:", styles['Subtitle']))
    job_summary = job_desc_text[:500] + "..." if len(job_desc_text) > 500 else job_desc_text
    elements.append(Paragraph(job_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add questions
    elements.append(Paragraph("Interview Questions:", styles['Subtitle']))
    elements.append(Spacer(1, 6))

    for i, question in enumerate(questions, 1):
        q_text = f"{i}. {question}"
        elements.append(Paragraph(q_text, styles['Normal']))

    # Build the PDF
    doc.build(elements)

    # Get the value from the buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def main():
    st.title("iHIRE")
    st.write("Upload a Job Description and Candidate CVs to Filter the Best Profiles and Generate Custom Interview Questions!")

    # Initialize session state for storing results
    if 'cv_similarities' not in st.session_state:
        st.session_state.cv_similarities = []
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'job_desc_text' not in st.session_state:
        st.session_state.job_desc_text = ""
    if 'questions_generated' not in st.session_state:
        st.session_state.questions_generated = {}
    if 'should_rerun' not in st.session_state:
        st.session_state.should_rerun = False

    # Check if we need to rerun from previous iteration
    if st.session_state.should_rerun:
        st.session_state.should_rerun = False
        st.rerun()

    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    num_questions = st.sidebar.slider("Number of questions to generate", 3, 15, 5)

    if not api_key:
        st.sidebar.warning("Please enter your Gemini API key to enable question generation")

    # File upload section
    st.header("Upload Files")
    job_desc_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
    cv_files = st.file_uploader("Upload Candidate CVs (PDF)", type="pdf", accept_multiple_files=True)

    # Match button
    if job_desc_file is not None and cv_files:
        if st.button("Match"):
            # Load the sentence transformer model
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            # Process job description
            job_desc_text = extract_text_from_pdf(job_desc_file)
            st.session_state.job_desc_text = job_desc_text
            job_desc_embedding = get_embedding(job_desc_text, model)

            # Process CVs and calculate similarities
            cv_similarities = []

            for cv_file in cv_files:
                cv_text = extract_text_from_pdf(cv_file)
                cv_embedding = get_embedding(cv_text, model)

                # Calculate similarity
                similarity = cosine_similarity(
                    [job_desc_embedding],
                    [cv_embedding]
                )[0][0]

                cv_similarities.append({
                    'filename': cv_file.name,
                    'similarity': similarity,
                    'text': cv_text
                })

            # Sort candidates by similarity
            cv_similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Keep only top 3 if there are more than 3
            if len(cv_similarities) > 3:
                cv_similarities = cv_similarities[:3]

            st.session_state.cv_similarities = cv_similarities
            st.session_state.show_results = True
            st.session_state.questions_generated = {}  # Reset questions when new matching is done

    # Display results
    if st.session_state.show_results and st.session_state.cv_similarities:
        st.header("Top 3 Matching Results")

        # Create checkboxes for selecting candidates
        selected_candidates = {}
        for idx, cv in enumerate(st.session_state.cv_similarities):
            similarity_percentage = cv['similarity'] * 100

            st.subheader(f"Candidate {idx + 1}: {cv['filename']}")
            st.write(f"Match Score: {similarity_percentage:.2f}%")

            with st.expander("View CV Content"):
                st.write(cv['text'])

            # Add checkbox for this candidate
            selected_candidates[cv['filename']] = st.checkbox(f"Select {cv['filename']} for question generation",
                                                              key=f"check_{cv['filename']}")

            # Show previously generated questions if they exist
            if cv['filename'] in st.session_state.questions_generated:
                st.subheader(f"Interview Questions for {cv['filename']}")
                for i, question in enumerate(st.session_state.questions_generated[cv['filename']], 1):
                    st.write(f"{i}. {question}")

                # Add download button for PDF
                try:
                    pdf_bytes = generate_pdf(cv['filename'], st.session_state.job_desc_text,
                                             st.session_state.questions_generated[cv['filename']])
                    pdf_filename = f"questions_{cv['filename'].replace(' ', '_')}.pdf"

                    st.markdown(create_download_link(pdf_bytes, pdf_filename), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

            st.markdown("---")

        # Generate questions button - only show if API key is provided
        if api_key:
            if st.button("Generate Questions for Selected Candidates"):
                questions_were_generated = False

                for cv in st.session_state.cv_similarities:
                    if selected_candidates.get(cv['filename'], False):
                        with st.spinner(f"Generating {num_questions} questions for {cv['filename']}..."):
                            context = f"Job Description:\n{st.session_state.job_desc_text}\n\nCandidate Profile:\n{cv['text']}"
                            questions = generate_questions_with_gemini(context, api_key, num_questions)
                            st.session_state.questions_generated[cv['filename']] = questions
                            questions_were_generated = True

                # Set flag to rerun on next iteration if questions were generated
                if questions_were_generated:
                    st.session_state.should_rerun = True
                    st.rerun()
        else:
            st.warning("Please enter your Gemini API key in the sidebar to generate questions")


if __name__ == "__main__":
    main()
