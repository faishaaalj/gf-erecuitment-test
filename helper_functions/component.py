# filename: function-app-project/helper_functions/ai_components.py
# Implements 3-stage scoring: AI Sub-scores -> Python Calc -> AI Reasoning

import logging
import json
import time
from typing import List, Optional, Tuple, Dict, Any
import openai # Ensure openai is imported if used directly for exceptions

# --- Local Module Imports ---
from . import services # Imports initialized clients
from .data_processing import format_job_details_for_prompt, format_candidate_details_for_prompt, _create_safe_dict_key # Import the key helper

logger = logging.getLogger(__name__)

def _call_openai_with_retry(messages, temperature=0.0, max_completion_tokens=1500, response_format=None, max_retries=3):
    """
    Wrapper to call OpenAI with retry logic for rate limiting and transient errors.
    """
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": services.openai_chat_deployment,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            response = services.openai_client.chat.completions.create(**kwargs)
            return response
        except openai.RateLimitError as e:
            wait_time = (2 ** attempt) + 1  # Exponential backoff: 1s, 3s, 5s
            logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s before retry. Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit error after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            # For other errors, don't retry
            logger.error(f"OpenAI API call failed (attempt {attempt+1}/{max_retries}): {e}")
            raise
    return None

# --- STAGE 1: AI SUB-SCORE EXTRACTION FUNCTION ---
# (Keep this function exactly as defined in the previous 'Refined Approach' update)
def _extract_ai_sub_scores(candidate_data: dict, job_details: dict) -> Optional[Dict[str, float]]:
    """
    Calls Azure OpenAI to extract sub-scores (0-100) for each criterion.
    Does NOT calculate the final weighted score. Returns None on failure.
    """
    logger.info("Starting AI sub-score extraction.")
    system_prompt_extract_template = """
Objective: You are a Detailed Criteria Assessor. Analyze the candidate against each specific job criterion provided and assign a sub-score representing the degree of match for that criterion ONLY.

Instructions:
1.  Read the 'Job Details & Weighted Criteria' and the 'Candidate Profile'.
2.  For each numbered criterion (Requirements, Responsibilities) and each 'Specific Criteria' item listed:
    - Determine the degree of match based ONLY on the Candidate Profile.
    - Assign a sub-score between 0.000 and 100.000 (3 decimal places).
    - IMPORTANT: If a criterion is marked MANDATORY (Weight: 100/100), the sub-score MUST be exactly 100.000 if met based on clear evidence in the profile/CV, or exactly 0.000 if not met or evidence is missing. No partial credit for mandatory items.
    - For non-mandatory criteria (Weight < 100), the sub-score should reflect the degree of alignment (e.g., 50.000 for partial match, 100.000 for perfect match, 0.000 for no match). Base this on the evidence found.
3.  Output ONLY a single JSON object containing these sub-scores, using the keys provided in the 'Output Schema'. Do not include weights or calculate a final score. No extra text.

Output Schema:
{{
  "sub_scores": {{
    "requirements": float (0.000-100.000),
    "responsibilities": float (0.000-100.000),
    {item_schema_placeholder}
  }}
}}
    """
    formatted_job_details_for_prompt = format_job_details_for_prompt(job_details)
    formatted_candidate_details = format_candidate_details_for_prompt(candidate_data)

    item_keys_list = []
    for item in job_details.get("items", []) or []:
        if item.get("scoreContribution", 0) > 0:
            category_key = _create_safe_dict_key(item.get("categoryName", "Unknown"))
            value_key = _create_safe_dict_key(item.get("value", "NotApplicable"))
            item_key_str = f"item_{category_key}_{value_key}"
            item_keys_list.append(f"\"{item_key_str}\": float (0.000-100.000)")

    item_schema_str = ",\n    ".join(item_keys_list) if item_keys_list else ""
    system_prompt_extract_final = system_prompt_extract_template.replace(
         '{item_schema_placeholder}', item_schema_str
    ).replace('{{','{').replace('}}','}')

    user_prompt = f"""
    Please perform the sub-score assessment according to the system prompt instructions.

    Job Details & Weighted Criteria:
    ---
    {formatted_job_details_for_prompt}
    ---

    Candidate Profile:
    ---
    {formatted_candidate_details}
    ---
    """

    result_json_str = ""
    try:
        logger.info("Calling OpenAI for sub-score extraction.")
        response = _call_openai_with_retry(
            messages=[{"role": "system", "content": system_prompt_extract_final}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_completion_tokens=1500,
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message.content:
            result_json_str = response.choices[0].message.content
            logger.debug(f"Raw sub-score response from AI: {result_json_str}")
            if result_json_str.strip().startswith('{') and result_json_str.strip().endswith('}'):
                result = json.loads(result_json_str)
                if "sub_scores" in result and isinstance(result["sub_scores"], dict):
                    logger.info("Sub-scores extracted successfully from AI response.")
                    validated_scores = {}
                    valid = True
                    for k, v in result["sub_scores"].items():
                        try:
                            score_val = float(v)
                            if not (0.0 <= score_val <= 100.0):
                                logger.error(f"Invalid sub-score value from AI for key '{k}': {v}. Out of range. Using 0.")
                                validated_scores[k] = 0.0
                                valid = False
                            else:
                                validated_scores[k] = score_val
                        except (ValueError, TypeError):
                             logger.error(f"Invalid sub-score type from AI for key '{k}': {v}. Expected float. Using 0.")
                             validated_scores[k] = 0.0
                             valid = False
                    return validated_scores # Return validated/corrected dict
                else:
                    logger.error(f"AI sub-score response missing 'sub_scores' dictionary. Response: {result_json_str}")
                    return None
            else:
                logger.error(f"AI sub-score response was not valid JSON structure. Response: {result_json_str}")
                return None
        else:
            # Handle content filtering or empty response
            finish_info = response.choices[0].finish_details if response.choices else None
            if finish_info and finish_info.get('type') == 'stop' and finish_info.get('stop') == 'content_filter':
                 logger.error("OpenAI sub-score response was filtered due to content policy AFTER generation.")
            else:
                logger.warning("OpenAI sub-score response was successful but contained no content.")
            return None

    except json.JSONDecodeError as json_err:
         logger.error(f"Failed to decode JSON sub-score response from AI: {json_err}. Raw Response: {result_json_str}")
         return None
    except openai.BadRequestError as e:
        if e.code == 'content_filter':
            logger.error(f"OpenAI sub-score prompt was filtered: {e.message}")
        else:
            logger.error(f"BadRequestError calling OpenAI for sub-scores: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error calling OpenAI for sub-scores: {e}", exc_info=True)
        return None


# --- STAGE 2: PYTHON SCORE CALCULATION ---
# (Modified to return score AND data needed for reasoning)
def _calculate_score_and_prepare_reasoning_data(
    ai_sub_scores: Dict[str, float],
    job_details: dict
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculates the final weighted score deterministically in Python based on AI sub-scores.
    Includes validation for mandatory criteria.
    Returns the final score and a dictionary containing data needed for reasoning generation.
    """
    logger.info("Starting Python score calculation.")
    total_score_numerator = 0.0
    total_weight_denominator = 0.0
    mandatory_failed_list = []
    criteria_details_for_reasoning = [] # Store details for the reasoning prompt

    # --- Process Requirements ---
    req_data = job_details.get('requirements', {})
    req_weight = float(req_data.get('scoreContribution', 0))
    req_name = "Persyaratan Umum"
    is_mandatory = (req_weight == 100)
    if req_weight > 0:
        sub_score = float(ai_sub_scores.get('requirements', 0.0))
        validated_sub_score = sub_score
        if is_mandatory and sub_score not in [0.0, 100.0]:
            logger.warning(f"AI sub-score {sub_score} for mandatory '{req_name}' overridden to 0.0.")
            validated_sub_score = 0.0
        if is_mandatory and validated_sub_score == 0.0:
            mandatory_failed_list.append(req_name)
        total_score_numerator += validated_sub_score * req_weight
        total_weight_denominator += req_weight
        criteria_details_for_reasoning.append({
            "name": req_name, "weight": int(req_weight),
            "sub_score": round(validated_sub_score, 3), "mandatory": is_mandatory
        })

    # --- Process Responsibilities ---
    resp_data = job_details.get('responsibilities', {})
    resp_weight = float(resp_data.get('scoreContribution', 0))
    resp_name = "Tanggung Jawab Utama"
    is_mandatory = (resp_weight == 100)
    if resp_weight > 0:
        sub_score = float(ai_sub_scores.get('responsibilities', 0.0))
        validated_sub_score = sub_score
        if is_mandatory and sub_score not in [0.0, 100.0]:
            logger.warning(f"AI sub-score {sub_score} for mandatory '{resp_name}' overridden to 0.0.")
            validated_sub_score = 0.0
        if is_mandatory and validated_sub_score == 0.0:
            mandatory_failed_list.append(resp_name)
        total_score_numerator += validated_sub_score * resp_weight
        total_weight_denominator += resp_weight
        criteria_details_for_reasoning.append({
            "name": resp_name, "weight": int(resp_weight),
            "sub_score": round(validated_sub_score, 3), "mandatory": is_mandatory
        })

    # --- Process Items ---
    for item in job_details.get("items", []) or []:
        item_weight = float(item.get('scoreContribution', 0))
        item_name_display = item.get("categoryName", "Kriteria Item Tidak Dikenal")
        if item_weight > 0:
            category_key = _create_safe_dict_key(item.get("categoryName", "Unknown"))
            value_key = _create_safe_dict_key(item.get("value", "NotApplicable"))
            lookup_key = f"item_{category_key}_{value_key}"
            sub_score = float(ai_sub_scores.get(lookup_key, 0.0))
            is_mandatory = (item_weight == 100)
            validated_sub_score = sub_score
            if is_mandatory and sub_score not in [0.0, 100.0]:
                logger.warning(f"AI sub-score {sub_score} for mandatory item '{item_name_display}' overridden to 0.0.")
                validated_sub_score = 0.0
            if is_mandatory and validated_sub_score == 0.0:
                mandatory_failed_list.append(item_name_display)
            total_score_numerator += validated_sub_score * item_weight
            total_weight_denominator += item_weight
            criteria_details_for_reasoning.append({
                "name": item_name_display, "weight": int(item_weight),
                "sub_score": round(validated_sub_score, 3), "mandatory": is_mandatory
            })

    # --- Calculate Final Score ---
    final_score = round(total_score_numerator / total_weight_denominator, 3) if total_weight_denominator > 0 else 0.0
    logger.info(f"Python Calculated: Numerator={total_score_numerator}, Denominator={total_weight_denominator}, Final Score={final_score}")

    # --- Prepare Data for Reasoning Generation ---
    reasoning_data = {
        "final_score": final_score,
        "mandatory_failed": mandatory_failed_list,
        "criteria_details": sorted(criteria_details_for_reasoning, key=lambda x: x['weight'], reverse=True) # Sort by importance
    }

    return final_score, reasoning_data


# --- STAGE 3: AI REASONING GENERATION ---
def _generate_ai_reasoning(
    candidate_data: dict,
    job_details: dict, # Pass original job details for context
    reasoning_data: Dict[str, Any] # Data from calculation step
) -> str:
    """
    Calls Azure OpenAI to generate a natural language reasoning string in Bahasa Indonesia,
    based on the calculated score and sub-score details.
    """
    logger.info("Starting AI reasoning generation.")
    system_prompt_reasoning = """
Objective: You are an expert HR Analyst. Your task is to generate a concise, descriptive reasoning text in Bahasa Indonesia explaining a candidate's calculated score.

Instructions:
1.  Review the 'Final Score', 'Mandatory Criteria Failures', and 'Criteria Details (Sub-Scores)' provided.
2.  Review the 'Candidate Profile' and 'Job Details' for context on what each criterion means.
3.  Generate the reasoning in Bahasa Indonesia with the following EXACT structure (use blank lines between sections):

[First line: Score statement]
Skor akhir kandidat adalah [EXACT SCORE AS PROVIDED - do not round or modify].

[Blank line, then Strengths section]
Kelebihan: [1-2 key strengths - criteria with high sub-scores, explaining WHY the candidate matched well based on their profile]

[Blank line, then Weaknesses section]  
Kekurangan: [1-2 key weaknesses - criteria with low sub-scores OR mandatory failures, explaining the mismatch. If mandatory criteria were failed, EXPLICITLY state which ones here]

[Blank line, then Conclusion section]
Kesimpulan: Secara keseluruhan, kandidat [sangat cocok / cukup cocok / kurang cocok] untuk peran ini berdasarkan kriteria yang dinilai.

4.  The reasoning should explain the score provided, NOT recalculate anything.
5.  Keep each section concise and professional.
6.  Output ONLY the reasoning text with the structure above. No JSON, no extra formatting.
7.  IMPORTANT: Always include blank lines (newlines) between each section for readability.
    """

    # Prepare context for the reasoning prompt
    formatted_job_details = format_job_details_for_prompt(job_details) # For context
    formatted_candidate_details = format_candidate_details_for_prompt(candidate_data) # For context

    # Use the exact final score value (as it appears in aiScore)
    final_score_display = reasoning_data['final_score']

    # Structure the input data clearly for the AI
    reasoning_input_context = f"""
Final Score Provided: {final_score_display}

Mandatory Criteria Failures: {', '.join(reasoning_data['mandatory_failed']) if reasoning_data['mandatory_failed'] else 'Tidak ada'}

Criteria Details (Sub-Scores assigned, out of 100):
{json.dumps(reasoning_data['criteria_details'], indent=2, ensure_ascii=False)}

Candidate Profile (for context):
---
{formatted_candidate_details}
---

Job Details (for context):
---
{formatted_job_details}
---

Generate the reasoning text in Bahasa Indonesia based *only* on the provided score and criteria details, using the profile and job details for context.
"""

    result_text = "Gagal menghasilkan alasan penilaian." # Default error message
    try:
        logger.info("Calling OpenAI for reasoning generation.")
        response = _call_openai_with_retry(
            messages=[
                {"role": "system", "content": system_prompt_reasoning},
                {"role": "user", "content": reasoning_input_context}
            ],
            temperature=0.2, # Allow a little creativity for natural language
            max_completion_tokens=500 # Adjust as needed for reasoning length
            # No response_format needed, expecting plain text
        )

        if response.choices and response.choices[0].message.content:
            result_text = response.choices[0].message.content.strip()
            logger.info("Reasoning text generated successfully.")
        else:
            # Handle content filtering or empty response
            finish_info = response.choices[0].finish_details if response.choices else None
            if finish_info and finish_info.get('type') == 'stop' and finish_info.get('stop') == 'content_filter':
                logger.error("OpenAI reasoning response was filtered due to content policy AFTER generation.")
                result_text = "Gagal menghasilkan alasan: Respons diblokir oleh filter konten."
            else:
                logger.warning("OpenAI reasoning response was successful but contained no content.")
                result_text = "Gagal menghasilkan alasan: AI tidak memberikan respons."

    except openai.BadRequestError as e:
         if e.code == 'content_filter':
            logger.error(f"OpenAI reasoning prompt was filtered: {e.message}")
            result_text = "Gagal menghasilkan alasan: Permintaan diblokir oleh filter konten."
         else:
            logger.error(f"BadRequestError calling OpenAI for reasoning: {e}", exc_info=True)
            result_text = f"Gagal menghasilkan alasan: Kesalahan permintaan ({e.message})."
    except Exception as e:
        logger.error(f"Error calling OpenAI for reasoning generation: {e}", exc_info=True)
        result_text = "Gagal menghasilkan alasan karena kesalahan internal."

    return result_text


# --- MAIN SCORING FUNCTION (Orchestrator - Updated for 3 Stages) ---
def get_ai_score(candidate_data: dict, job_details: dict) -> Tuple[float, str]:
    """
    Orchestrates the 3-stage scoring process:
    1. AI extracts sub-scores.
    2. Python validates, calculates final score, prepares reasoning data.
    3. AI generates natural language reasoning based on calculation results.
    """
    logger.info("Starting 3-stage scoring process for candidate.")

    # Stage 1: AI Extracts Sub-Scores
    ai_sub_scores = _extract_ai_sub_scores(candidate_data, job_details)
    if ai_sub_scores is None:
        logger.error("Stage 1 failed: Could not extract sub-scores from AI. Returning score 0.")
        return 0.0, "Gagal mengekstrak sub-skor dari AI."

    logger.info("Stage 1 successful: AI Sub-scores extracted.")
    logger.debug(f"AI Sub-scores content: {ai_sub_scores}")

    # Stage 2: Python Calculates Final Score and Prepares Reasoning Data
    try:
        final_score, reasoning_data_for_ai = _calculate_score_and_prepare_reasoning_data(ai_sub_scores, job_details)
        logger.info(f"Stage 2 successful: Python calculation complete. Final score: {final_score}")
    except Exception as e:
        logger.error(f"Stage 2 failed: Error during Python calculation: {e}", exc_info=True)
        return 0.0, "Terjadi kesalahan saat menghitung skor akhir."

    # Stage 3: AI Generates Natural Language Reasoning
    try:
        final_reasoning = _generate_ai_reasoning(candidate_data, job_details, reasoning_data_for_ai)
        logger.info("Stage 3 successful: AI reasoning generated.")
        # Optional: Add validation if reasoning is empty/failed
        if not final_reasoning or "Gagal" in final_reasoning:
             logger.warning("AI reasoning generation failed or returned an error message.")
             # Fallback to a simpler reasoning? Or return the error message?
             # For now, return the error message from the AI function.

        return final_score, final_reasoning
    except Exception as e:
        logger.error(f"Stage 3 failed: Error during AI reasoning generation: {e}", exc_info=True)
        # Return the calculated score but indicate reasoning failure
        return final_score, "Terjadi kesalahan saat menghasilkan alasan penilaian."


# --- summarize_candidate_for_table and generate_embedding ---
# (Keep these as they are used by ranked-candidates and orchestrators/activities)
def summarize_candidate_for_table(candidate_summary_input: dict) -> dict:
    # ... (Keep previous robust version) ...
    logger.info("Starting candidate summarization for table.")
    system_prompt_summary = """
    You are an efficient HR assistant. Your task is to summarize a candidate's profile into concise data points for a table display.
    Based on the provided candidate data, extract the following information and provide the response ONLY in JSON format. Use "N/A" if information is missing.
    - "experience": A summary of total work experience (e.g., "3 years (Sales FMCG)"). Focus on most relevant/recent.
    - "skill": A comma-separated list of the 4-5 most relevant top skills based on the CV content and experience.
    - "location": The candidate's location (City only from input, if available).
    - "lastWork": The name of the company where the candidate last worked (most recent).
    - "education": A summary of the latest or highest education (e.g., "S1 Management - Universitas Padjadjaran").
    Ensure the output is a single, valid JSON object.
    """
    serializable_input = json.loads(json.dumps(candidate_summary_input, default=str))
    user_content_str = json.dumps(serializable_input, ensure_ascii=False)
    result_json_str = ""
    try:
        logger.info("Calling OpenAI for summarization.")
        response = _call_openai_with_retry(
            messages=[{"role": "system", "content": system_prompt_summary}, {"role": "user", "content": user_content_str}],
            max_completion_tokens=300,
            response_format={"type": "json_object"}
        )
        if response.choices and response.choices[0].message.content:
            result_json_str = response.choices[0].message.content.strip()
            if result_json_str.startswith('{') and result_json_str.endswith('}'):
                logger.info("Summarization successful.")
                return json.loads(result_json_str)
            else: logger.error(f"AI summary did not return valid JSON structure: {result_json_str}")
        else:
            finish_info = response.choices[0].finish_details if response.choices else None
            if finish_info and finish_info.get('type') == 'stop' and finish_info.get('stop') == 'content_filter': logging.error("OpenAI summary response was filtered.")
            else: logging.warning("OpenAI summary response was empty.")
    except json.JSONDecodeError as json_err: logger.error(f"Failed to decode JSON summary: {json_err}. Raw: {result_json_str}")
    except openai.BadRequestError as e:
         if e.code == 'content_filter': logging.error(f"OpenAI summary prompt was filtered: {e.message}")
         else: logger.error(f"BadRequestError during summarization: {e}", exc_info=True)
    except Exception as e: logger.error(f"Error summarizing candidate: {e}", exc_info=True)
    logger.warning("Returning default N/A structure for summarization.")
    return {"experience": "N/A", "skill": "N/A", "location": "N/A", "lastWork": "N/A", "education": "N/A"}

def _find_existing_application(candidate_id: str, job_id: str) -> Optional[str]:
    """Queries AI Search to find the applicationId if an application already exists."""
    try:
        search_results = services.search_client.search(
            search_text="", # No text search needed, just filter
            filter=f"candidateId eq '{candidate_id}' and jobId eq '{job_id}'",
            select="applicationId", # Only need the ID
            top=1 # Expect only one or zero
        )
        first_result = next(search_results, None)
        if first_result:
            return first_result.get("applicationId")
        return None
    except Exception as e:
        logging.error(f"Error searching for existing application for candidate {candidate_id}, job {job_id}: {e}", exc_info=True)
        return None # Treat errors as 'not found' to allow submission


def generate_embedding(text: str) -> Optional[List[float]]:
    # ... (Keep previous robust version) ...
    if not text:
        logger.warning("Attempted to generate embedding for empty text.")
        return None
    try:
        logger.debug("Generating embedding.")
        response = services.openai_client.embeddings.create(
            model=services.openai_embedding_deployment, input=text
        )
        logger.debug("Embedding generated successfully.")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None


def extract_candidate_industries(work_experience: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Extracts broad industry categories from candidate's work experience.
    Uses AI to analyze company names and job descriptions to classify industries.
    
    Returns:
        Tuple containing:
        - List of industries per work experience (same order, aligned with input)
        - List of unique industry names for flat industries field
    
    Example:
        (["Oil & Gas", "FMCG", "Banking"], ["Oil & Gas", "FMCG", "Banking"])
    """
    if not work_experience or not isinstance(work_experience, list):
        logger.warning("No work experience provided for industry extraction.")
        return [], []
    
    logger.info(f"Extracting industries from {len(work_experience)} work experience entries.")
    
    # Prepare work experience data for the AI prompt
    experience_summary = []
    for i, exp in enumerate(work_experience, 1):
        company = exp.get('companyName', 'Unknown Company')
        position = exp.get('jobTitle', exp.get('position', 'Unknown Position'))  # Try jobTitle first
        description = exp.get('description', exp.get('jobDescription', ''))  # Try both field names
        # Truncate description to keep prompt manageable
        desc_snippet = description[:200] if description else ''
        experience_summary.append(
            f"{i}. Company: {company}\n   Position: {position}\n   Description: {desc_snippet}"
        )
    
    experience_text = "\n\n".join(experience_summary)
    
    system_prompt = """
You are an industry classification expert. Your task is to analyze a candidate's work experience and classify each job into a broad industry category.

Instructions:
1. Review the provided work experience entries (company names, positions, job descriptions).
2. For EACH experience entry (numbered 1, 2, 3...), determine the primary industry sector based on:
   - Company name (e.g., "Bank Mandiri" → Banking)
   - Job description content
   - Position title
3. Use standard industry classifications (e.g., Banking, Oil & Gas, E-commerce, FMCG, Technology, Healthcare, Manufacturing, Retail, Education, Hospitality, Telecommunications, etc.)
4. If you cannot determine the industry, use "Other"
5. Return a JSON object with:
   - "per_job_industries": An array of industries, one for each job IN THE SAME ORDER as provided
   - "unique_industries": A unique list of all industries found (for summary)

Output Schema:
{
  "per_job_industries": ["Industry1", "Industry2", "Industry3", ...],
  "unique_industries": ["Industry1", "Industry2", ...]
}
"""
    
    user_prompt = f"""
Please analyze the following work experience entries and classify each into its primary industry:

{experience_text}

Return the industries as a JSON object with both per-job and unique lists.
"""
    
    result_json_str = ""
    try:
        logger.info("Calling OpenAI for industry extraction.")
        response = _call_openai_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent categorization
            max_completion_tokens=500,  # Increased for per-job classifications
            response_format={"type": "json_object"}
        )
        
        if response.choices and response.choices[0].message.content:
            result_json_str = response.choices[0].message.content.strip()
            logger.debug(f"Raw industry extraction response: {result_json_str}")
            
            if result_json_str.startswith('{') and result_json_str.endswith('}'):
                result = json.loads(result_json_str)
                per_job_industries = result.get('per_job_industries', [])
                unique_industries = result.get('unique_industries', [])
                
                # Validate per_job_industries
                if isinstance(per_job_industries, list):
                    # Ensure we have the same number as work experience entries
                    if len(per_job_industries) != len(work_experience):
                        logger.warning(f"Mismatch: {len(per_job_industries)} industries vs {len(work_experience)} jobs. Padding with 'Other'.")
                        # Pad or truncate to match
                        while len(per_job_industries) < len(work_experience):
                            per_job_industries.append("Other")
                        per_job_industries = per_job_industries[:len(work_experience)]
                    
                    # Clean and validate each entry
                    validated_per_job = []
                    for ind in per_job_industries:
                        if isinstance(ind, str) and ind.strip():
                            validated_per_job.append(ind.strip())
                        else:
                            validated_per_job.append("Other")
                    
                    # Validate unique_industries
                    validated_unique = []
                    if isinstance(unique_industries, list):
                        for ind in unique_industries:
                            if isinstance(ind, str) and ind.strip() and ind.strip() != "Other":
                                validated_unique.append(ind.strip())
                    
                    # If no unique list provided, derive from per_job
                    if not validated_unique:
                        validated_unique = list(set(ind for ind in validated_per_job if ind != "Other"))
                    
                    logger.info(f"Successfully extracted industries. Per-job: {validated_per_job}, Unique: {validated_unique}")
                    return validated_per_job, validated_unique
                else:
                    logger.error(f"'per_job_industries' field is not a list: {per_job_industries}")
                    return ["Other"] * len(work_experience), []
            else:
                logger.error(f"AI industry response was not valid JSON: {result_json_str}")
                return ["Other"] * len(work_experience), []
        else:
            finish_info = response.choices[0].finish_details if response.choices else None
            if finish_info and finish_info.get('type') == 'stop' and finish_info.get('stop') == 'content_filter':
                logger.error("OpenAI industry extraction response was filtered due to content policy.")
            else:
                logger.warning("OpenAI industry extraction response was empty.")
            return ["Other"] * len(work_experience), []
    
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON industry response: {json_err}. Raw: {result_json_str}")
        return ["Other"] * len(work_experience), []
    except openai.BadRequestError as e:
        if e.code == 'content_filter':
            logger.error(f"OpenAI industry extraction prompt was filtered: {e.message}")
        else:
            logger.error(f"BadRequestError during industry extraction: {e}", exc_info=True)
        return ["Other"] * len(work_experience), []
    except Exception as e:
        logger.error(f"Error extracting candidate industries: {e}", exc_info=True)
        return ["Other"] * len(work_experience), []