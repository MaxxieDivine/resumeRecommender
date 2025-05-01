# job_matcher.py
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import logging

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify the Sentence Transformer model
# 'all-MiniLM-L6-v2' is a good balance of speed and performance.
MODEL_NAME = 'all-MiniLM-L6-v2'
# Number of top job matches to return
TOP_N = 10

# --- Core Functions ---

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extracts text content from a given PDF file.

    Args:
        pdf_path: Path to the PDF resume file.

    Returns:
        A string containing the extracted text, or None if an error occurs
        or the file doesn't exist.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return None

    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            logging.info(f"Opened PDF: {pdf_path}. Pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                else:
                    logging.warning(f"Could not extract text from page {i+1} of {pdf_path}")
            # Handle cases where text extraction might yield None or empty strings per page
            if not full_text.strip():
                 logging.warning(f"No text could be extracted from the PDF: {pdf_path}")
                 return None
            logging.info(f"Successfully extracted text from {pdf_path}.")
        return full_text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def load_jobs_from_csv(csv_path: str, title_col: str = 'title', desc_col: str = 'description', id_col: str | None = 'jobLink') -> pd.DataFrame | None:
    """
    Loads job postings from a CSV file into a pandas DataFrame.
    Assigns a unique 'job_id' if an ID column isn't specified or found.

    Args:
        csv_path: Path to the CSV file containing job postings.
        title_col: Name of the column containing the job title. Defaults to 'title'.
        desc_col: Name of the column containing the job description. Defaults to 'description'.
        id_col: Optional name of the column containing a unique job ID. Defaults to 'jobLink'.

    Returns:
        A pandas DataFrame with job data, or None if an error occurs.
    """
    if not os.path.exists(csv_path):
        logging.error(f"Error: CSV file not found at {csv_path}")
        return None

    try:
        # Attempt to read CSV, handling potential parsing errors
        df = pd.read_csv(csv_path, on_bad_lines='warn') # Warn about bad lines instead of failing outright
        logging.info(f"Successfully loaded CSV: {csv_path}. Shape: {df.shape}")

        # --- Validate required columns ---
        required_cols = [title_col, desc_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"CSV missing required columns based on configuration: {', '.join(missing_cols)}. Available columns: {list(df.columns)}")
            return None

        # --- Handle Job ID ---
        actual_id_col_name = None
        if id_col and id_col in df.columns:
            actual_id_col_name = id_col # Use the specified and existing column
            logging.info(f"Using existing column '{id_col}' as source for 'job_id'.")
        else:
            if id_col:
                logging.warning(f"Specified id_col '{id_col}' not found in CSV columns: {list(df.columns)}. Generating sequential IDs instead.")
            else:
                 logging.info("No id_col specified. Generating sequential IDs.")
            # Generate sequential IDs if id_col is None or not found
            df['job_id_generated'] = range(len(df))
            actual_id_col_name = 'job_id_generated' # Use the generated column

        # --- Select and Rename Columns ---
        # Keep only necessary columns and standardize names
        # Ensure the actual_id_col_name is included
        columns_to_keep = [actual_id_col_name, title_col, desc_col]
        df = df[columns_to_keep].copy()

        # Rename columns to standard names used internally
        df.rename(columns={
            title_col: 'title',
            desc_col: 'job_description',
            actual_id_col_name: 'job_id' # Rename the selected ID column to 'job_id'
        }, inplace=True)


        # --- Handle Missing/Invalid Data ---
        initial_rows = len(df)
        # Drop rows where title or description is missing or not a string
        df.dropna(subset=['title', 'job_description'], inplace=True)
        df = df[df['title'].apply(lambda x: isinstance(x, str))]
        df = df[df['job_description'].apply(lambda x: isinstance(x, str))]

        # Convert job_id to string to handle potential mixed types from CSV
        df['job_id'] = df['job_id'].astype(str)

        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logging.warning(f"Dropped {rows_dropped} rows due to missing/invalid titles or descriptions.")

        if df.empty:
            logging.error("No valid jobs found in the CSV after cleaning (checking for non-empty string titles and descriptions).")
            return None

        return df

    except pd.errors.EmptyDataError:
        logging.error(f"Error: CSV file {csv_path} is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading or processing CSV {csv_path}: {e}")
        return None

def calculate_similarity(resume_text: str, jobs_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates cosine similarity between resume text and job descriptions.

    Args:
        resume_text: The extracted text from the resume.
        jobs_df: DataFrame containing job postings with 'job_description' column.

    Returns:
        The jobs_df DataFrame with an added 'similarity_score' column,
        or None if an error occurs.
    """
    if not resume_text or not isinstance(resume_text, str):
        logging.error("Cannot calculate similarity: Resume text is empty or invalid.")
        return None
    if jobs_df is None or jobs_df.empty:
        logging.error("Cannot calculate similarity: Jobs DataFrame is empty or None.")
        return None
    if 'job_description' not in jobs_df.columns:
        logging.error("Cannot calculate similarity: 'job_description' column missing in jobs DataFrame.")
        return None
    if not pd.api.types.is_string_dtype(jobs_df['job_description']):
         logging.warning("Job description column contains non-string data. Attempting to convert to string.")
         jobs_df['job_description'] = jobs_df['job_description'].astype(str)


    try:
        logging.info(f"Loading sentence transformer model: {MODEL_NAME}...")
        # Load the pre-trained sentence transformer model
        model = SentenceTransformer(MODEL_NAME)
        logging.info("Model loaded successfully.")

        # --- Generate Embeddings ---
        logging.info("Generating embedding for the resume...")
        # Ensure resume_text is treated as a single document
        resume_embedding = model.encode([resume_text], show_progress_bar=False, normalize_embeddings=True) # Normalize for cosine sim
        logging.info("Resume embedding generated.")

        logging.info("Generating embeddings for job descriptions...")
        # Ensure job descriptions are processed correctly
        job_descriptions = jobs_df['job_description'].tolist()
        job_embeddings = model.encode(job_descriptions, show_progress_bar=True, normalize_embeddings=True) # Normalize for cosine sim
        logging.info("Job description embeddings generated.")

        # --- Calculate Cosine Similarity ---
        logging.info("Calculating cosine similarities...")
        # resume_embedding is shape (1, embedding_dim)
        # job_embeddings is shape (num_jobs, embedding_dim)
        # similarities will be shape (1, num_jobs)
        # Since embeddings are normalized, dot product is equivalent to cosine similarity
        similarities = resume_embedding @ job_embeddings.T

        # Add scores to the DataFrame. Flatten the similarities array.
        jobs_df['similarity_score'] = similarities[0]
        logging.info("Similarity scores calculated and added to DataFrame.")

        return jobs_df

    except Exception as e:
        logging.error(f"Error during embedding or similarity calculation: {e}")
        # Optionally log traceback for more details
        # import traceback
        # logging.error(traceback.format_exc())
        return None

def rank_jobs(jobs_df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Ranks jobs based on similarity score in descending order.

    Args:
        jobs_df: DataFrame with 'similarity_score' column.
        top_n: The number of top results to return.

    Returns:
        A DataFrame containing the top N ranked jobs.
    """
    if jobs_df is None or 'similarity_score' not in jobs_df.columns:
        logging.warning("Cannot rank jobs: DataFrame is None or missing 'similarity_score'. Returning empty DataFrame.")
        return pd.DataFrame() # Return empty DataFrame

    # Sort by score and select top N
    ranked_jobs = jobs_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    logging.info(f"Ranked jobs and selected top {min(top_n, len(ranked_jobs))}.")
    return ranked_jobs

def export_results(results_df: pd.DataFrame, output_path: str) -> bool:
    """
    Exports the ranked job results to a CSV file.

    Args:
        results_df: DataFrame containing the ranked job matches.
        output_path: Path to save the output CSV file.

    Returns:
        True if export was successful, False otherwise.
    """
    if results_df is None or results_df.empty:
        logging.warning("No results to export.")
        return False
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        # Select columns to export - keep it clean
        columns_to_export = ['job_id', 'title', 'similarity_score', 'job_description']
        export_df = results_df[[col for col in columns_to_export if col in results_df.columns]]

        export_df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Successfully exported results to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error exporting results to CSV {output_path}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration - Replace with your file paths ---
    # Example Usage:
    # Ensure you have a PDF resume and a CSV with job postings.
    # The CSV should ideally have columns matching the defaults below,
    # otherwise, change the variables to match your CSV.
    RESUME_PDF_PATH = 'max_locher_resume.pdf'      # <--- CHANGE THIS
    JOBS_CSV_PATH = 'jobExportsApify.csv'          # <--- CHANGE THIS
    OUTPUT_CSV_PATH = 'output/top_job_matches.csv'   # <--- CHANGE THIS (optional)

    # --- Column names in your jobs CSV (Update if different from the example format) ---
    JOB_TITLE_COLUMN = 'title'           # Default based on the provided example
    JOB_DESC_COLUMN = 'description'      # Default based on the provided example
    JOB_ID_COLUMN = 'jobLink'            # Default based on the provided example (use None to generate sequential IDs)

    logging.info("--- Starting Job Matcher ---")

    # 1. Extract Resume Text
    logging.info(f"Attempting to extract text from: {RESUME_PDF_PATH}")
    resume_text = extract_text_from_pdf(RESUME_PDF_PATH)

    if not resume_text:
        logging.error("Failed to extract text from resume. Exiting.")
        exit(1) # Exit if resume processing fails
    logging.info("Resume text extracted successfully.")
    # logging.debug(f"Resume Text (first 100 chars): {resume_text[:100]}...") # Uncomment for debugging

    # 2. Load Job Postings
    logging.info(f"Attempting to load jobs from: {JOBS_CSV_PATH}")
    logging.info(f"Using columns - Title: '{JOB_TITLE_COLUMN}', Description: '{JOB_DESC_COLUMN}', ID: '{JOB_ID_COLUMN}'")
    jobs_data = load_jobs_from_csv(
        JOBS_CSV_PATH,
        title_col=JOB_TITLE_COLUMN,
        desc_col=JOB_DESC_COLUMN,
        id_col=JOB_ID_COLUMN
    )

    if jobs_data is None:
        logging.error("Failed to load or process job postings. Exiting.")
        exit(1) # Exit if job loading fails
    logging.info(f"Job postings loaded and processed successfully. Found {len(jobs_data)} valid jobs.")

    # 3. Calculate Similarity
    logging.info("Calculating similarities between resume and jobs...")
    jobs_with_scores = calculate_similarity(resume_text, jobs_data)

    if jobs_with_scores is None:
        logging.error("Failed to calculate similarities. Exiting.")
        exit(1) # Exit if similarity calculation fails
    logging.info("Similarity calculation complete.")

    # 4. Rank Jobs
    logging.info(f"Ranking jobs and selecting top {TOP_N}...")
    top_matches = rank_jobs(jobs_with_scores, top_n=TOP_N)

    if top_matches.empty:
         logging.warning("No matching jobs found after ranking.")
    else:
        logging.info("Top job matches identified:")
        # Display results in the console (optional) - showing only key columns
        print("\n--- Top Job Matches ---")
        print(top_matches[['job_id', 'title', 'similarity_score']].round({'similarity_score': 4}).to_string(index=False))
        print("-----------------------\n")


    # 5. Export Results
    logging.info(f"Attempting to export results to: {OUTPUT_CSV_PATH}")
    export_successful = export_results(top_matches, OUTPUT_CSV_PATH)

    if export_successful:
        logging.info("Job matching process completed successfully.")
    else:
        logging.warning("Job matching process completed, but failed to export results.")

    logging.info("--- Job Matcher Finished ---")
