# Semantic Job Matcher with LLM Explanation

## Product Management Overview

### What is this?

This project is a Python application designed to automate and enhance the job searching process. It takes a user's resume (in PDF format) and a list of job postings (in CSV format) as input. Using semantic analysis and optionally a local Large Language Model (LLM), it outputs a ranked list of the most relevant job opportunities based on the meaning and context within the resume and job descriptions, along with an AI-generated explanation for the top matches.

### Why was this built? (The Problem & Value)

Job searching can be a time-consuming and often frustrating process. Manually sifting through hundreds of job postings, many of which only match based on superficial keywords, is inefficient. Traditional keyword-based search tools often miss relevant opportunities simply because the terminology used in the job description doesn't perfectly align with the resume, even if the underlying skills and experiences are a strong match.

This tool addresses these pain points by:

1.  **Prioritizing Relevance over Keywords:** It leverages sentence transformers to understand the *semantic meaning* of the text in both the resume and job descriptions. This allows it to identify strong matches based on conceptual similarity, even if the exact phrasing differs, leading to higher quality, more relevant results.
2.  **Increasing Efficiency:** Automating the initial screening process saves significant time compared to manually reviewing every job posting. Users can focus their energy on the opportunities that are most likely to be a good fit.
3.  **Providing Deeper Insights (with LLM):** The integration with a local LLM (like Llama 3 via Ollama) adds a layer of explainability. Instead of just seeing a similarity score, users get a concise, AI-generated summary of *why* a specific job is considered a good match, highlighting key skill and experience overlaps. This aids validation and helps tailor applications.
4.  **Maintaining Privacy (with Local LLM):** By using a locally run LLM via Ollama, sensitive resume data remains on the user's machine, addressing privacy concerns associated with uploading personal documents to third-party services.
5.  **Focusing the Search:** By surfacing the most semantically relevant roles first, it helps users concentrate their efforts on applications with the highest potential for success.

Essentially, this tool aims to make the job search smarter, faster, and more insightful by applying modern AI techniques to understand the *true* alignment between a candidate and a role.

## Technical Overview

### How it Works

The application follows these core steps:

1.  **PDF Resume Parsing:**
    * The script uses the `pdfplumber` library to open the user-provided PDF resume file.
    * It iterates through each page, extracts the text content, and concatenates it into a single string.
    * Basic error handling is included for file not found or text extraction issues.

2.  **Job Postings CSV Loading:**
    * It uses the `pandas` library to load the job postings from the specified CSV file.
    * It expects configurable column names for the Job Title, Job Description, and an optional Job ID (defaulting to `title`, `description`, and `jobLink` based on the provided sample).
    * It performs basic cleaning: drops rows with missing titles or descriptions and ensures required columns are present. If a specific Job ID column isn't found or provided, it generates sequential IDs.

3.  **Semantic Embedding Generation:**
    * The core of the semantic matching relies on the `sentence-transformers` library.
    * It loads a pre-trained model (e.g., `all-MiniLM-L6-v2`), which is efficient and effective for generating sentence/paragraph embeddings.
    * The entire resume text is encoded into a single vector embedding (normalized).
    * Each job description in the loaded DataFrame is also encoded into its own vector embedding (normalized).

4.  **Similarity Calculation:**
    * The script uses `sklearn.metrics.pairwise.cosine_similarity` to calculate the similarity between the resume embedding and *each* job description embedding. Since the embeddings are normalized, this is efficiently calculated as a matrix multiplication (dot product).
    * The resulting similarity scores (ranging from -1 to 1, typically 0 to 1 for positive correlations) represent the semantic closeness between the resume and each job.
    * These scores are added as a new column (`similarity_score`) to the jobs DataFrame.

5.  **Ranking:**
    * The jobs DataFrame is sorted in descending order based on the `similarity_score`.
    * The top N jobs (default is 10) are selected.

6.  **LLM Explanation Generation (Optional):**
    * If enabled (`ENABLE_LLM_EXPLANATION = True`), the script iterates through the top N matched jobs.
    * For each job, it constructs a specific prompt containing both the full resume text and the job description.
    * It sends this prompt via an HTTP POST request to the configured local Ollama API endpoint (`http://localhost:11434/api/generate` by default).
    * The request specifies the local LLM model to use (e.g., `llama3`).
    * It uses the `requests` library to handle the API call and includes error handling for connection issues, timeouts, and bad responses (like 404 if the model isn't found).
    * The LLM's generated text response (the explanation) is parsed from the JSON response.
    * These explanations are added as a new column (`llm_explanation`) to the top matches DataFrame.

7.  **Results Output:**
    * The final DataFrame containing the top N jobs, their similarity scores, and optionally the LLM explanations, is displayed in the console.
    * The same DataFrame is exported to a CSV file (e.g., `output/top_job_matches_explained.csv`) for persistent storage and review.

### Key Libraries Used

* `pdfplumber`: For extracting text from PDF files.
* `pandas`: For data manipulation and handling the CSV data.
* `sentence-transformers`: For generating semantic text embeddings.
* `scikit-learn`: For calculating cosine similarity.
* `numpy`: For numerical operations (often used implicitly by other libraries).
* `requests`: For making API calls to the local Ollama LLM server.
* `logging`: For providing informative status messages and error reporting.
* `json`: For handling JSON data in API requests/responses.

### Setup and Usage

(Refer to the `instructions_md` or `ollama_instructions_md` artifacts for detailed setup and execution steps).

1.  Install required libraries: `pip install -r requirements.txt`
2.  Ensure Ollama is running with the specified model (`llama3` by default) if using the explanation feature.
3.  Configure file paths (`RESUME_PDF_PATH`, `JOBS_CSV_PATH`, `OUTPUT_CSV_PATH`) and CSV column names (`JOB_TITLE_COLUMN`, etc.) in the script's `if __name__ == "__main__":` block.
4.  Run the script: `python job_matcher.py`
