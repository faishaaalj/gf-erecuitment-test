# filename: function-app-project/function_app.py
# Refactored to use Azure Durable Functions

import azure.functions as func
import azure.durable_functions as df
import logging
import json
import datetime
import uuid
from typing import List, Dict, Any, Optional

from helper_functions import services
from helper_functions.data_processing import format_candidate_details_for_prompt, format_job_details_for_prompt, _create_safe_dict_key, clean_html
# from helper_functions.data_models import get_dummy_job_details
from helper_functions.job_api_client import get_job_details
from helper_functions.component import _find_existing_application, get_ai_score, summarize_candidate_for_table, generate_embedding, extract_candidate_industries

bp = df.Blueprint()

def _prepare_field_for_index(value, field_name=""):
    """Helper to ensure field values are compatible with AI Search schema.
    Converts arrays to comma-separated strings for fields expecting Edm.String.
    """
    if value is None:
        return None
    # If it's a list/array and should be a string, join it
    if isinstance(value, list) and not all(isinstance(x, dict) for x in value):
        # Simple list of strings/values - join them
        return ", ".join(str(v) for v in value if v)
    return value

def _parse_gpa(gpa_value):
    """Helper to parse GPA values that may be in format '3.87/4.00', '3.87', or '3,71' (comma as decimal).
    Returns float value or None if invalid.
    """
    if gpa_value is None:
        return None
    
    # If already a number, return it
    if isinstance(gpa_value, (int, float)):
        return float(gpa_value)
    
    # If string, try to parse
    if isinstance(gpa_value, str):
        gpa_str = gpa_value.strip()
        if not gpa_str:
            return None
        
        # Replace comma with dot for decimal separator (e.g., '3,71' -> '3.71')
        gpa_str = gpa_str.replace(',', '.')
        
        # Check if it contains '/' (e.g., '3.87/4.00')
        if '/' in gpa_str:
            try:
                # Extract the numerator part before '/'
                numerator = gpa_str.split('/')[0].strip()
                return float(numerator)
            except (ValueError, IndexError):
                logging.warning(f"Failed to parse GPA value: {gpa_value}")
                return None
        else:
            # Try to convert directly
            try:
                return float(gpa_str)
            except ValueError:
                logging.warning(f"Failed to parse GPA value: {gpa_value}")
                return None
    
    return None

def _encode_cv_url(cv_url):
    """Helper to encode CV URL by replacing spaces with %20.
    Handles URLs that may contain spaces in filenames.
    """
    if not cv_url or not isinstance(cv_url, str):
        return cv_url
    
    # Only encode the path part, not the entire URL
    from urllib.parse import quote
    
    # Split URL into base and path
    if '://' in cv_url:
        # Find the position after the domain
        parts = cv_url.split('/', 3)  # Split into protocol, empty, domain, path
        if len(parts) >= 4:
            protocol_and_domain = '/'.join(parts[:3])  # e.g., 'https://career.garudafood.co.id'
            path = parts[3]  # e.g., 'Files/Resume/20251111553679Curriculum Vitae.pdf'
            # Encode only the path part, keeping '/' as is
            encoded_path = quote(path, safe='/=')
            return f"{protocol_and_domain}/{encoded_path}"
    
    # Fallback: simple space replacement if URL structure is unusual
    return cv_url.replace(' ', '%20')

@bp.activity_trigger(input_name="jobId")
async def GetJobDetailsActivity(jobId: str) -> Optional[dict]:
    """
    Activity function to get job details from the live API.
    Handles auth token caching and retrieval.
    """
    logging.info(f"Activity: Getting job details for {jobId} from live API.")

    details = await get_job_details(jobId)
    if not details:
        logging.warning(f"Activity: Job details not found for {jobId}.")
    return details

@bp.route(route="apply_job")
@bp.durable_client_input(client_name="client")
async def http_submit_cv_starter(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    """
    HTTP endpoint to start the application processing orchestration.
    Validates input and starts the background task.
    """
    logging.info('HTTP Starter: submit-cv request received.')
    try:
        candidate_data = req.get_json()
        # Basic validation - check if fields exist and are not empty
        required_fields = ["jobId", "cv", "candidateId"]
        if not all(k in candidate_data and candidate_data.get(k) for k in required_fields):
             logging.warning("HTTP Starter: Missing or empty required fields in submit-cv request.")
             return func.HttpResponse("Missing or empty required fields: jobId, cv, candidateId", status_code=400)

        instance_id = await client.start_new(
            orchestration_function_name="ApplicationProcessingOrchestrator",
            instance_id=None,
            client_input=candidate_data
        )

        logging.info(f"HTTP Starter: Started orchestration with ID = '{instance_id}'.")

        return client.create_check_status_response(req, instance_id)

    except ValueError:
        return func.HttpResponse("Invalid JSON format in request body.", status_code=400)
    except Exception as e:
        logging.error(f"HTTP Starter Error (submit-cv): {e}", exc_info=True)
        return func.HttpResponse("Internal server error.", status_code=500)


# Orchestrator for Application Processing
@bp.orchestration_trigger(context_name="context")
def ApplicationProcessingOrchestrator(context: df.DurableOrchestrationContext):
    """
    Orchestrates application processing: Find Existing -> Analyze CV -> Get Job Details -> Perform Scoring -> Generate Embeddings -> Index.
    """
    orchestration_input: Dict[str, Any] = context.get_input()
    instance_id = context.instance_id
    logging.info(f"Orchestrator ({instance_id}): Starting application processing.")

    candidate_id = orchestration_input.get('candidateId')
    job_id = orchestration_input.get('jobId')
    cv_url = orchestration_input.get('cv')

    if not all([candidate_id, job_id, cv_url]):
        error_msg = f"Orchestrator ({instance_id}): Invalid input."
        logging.error(error_msg)
        return {"status": "Failed", "error": error_msg}

    try:
        # Step 1: Check for Existing Application
        existing_application_id = yield context.call_activity("FindExistingApplicationActivity", {"candidateId": candidate_id, "jobId": job_id})
        application_id = existing_application_id or str(uuid.uuid4())
        operation_type = "updated" if existing_application_id else "created"
        logging.info(f"Orchestrator ({instance_id}): Application ID: {application_id} (Operation: {operation_type})")

        # Step 2: Analyze CV
        logging.info(f"Orchestrator ({instance_id}): Calling AnalyzeCvActivity.")
        cv_content = yield context.call_activity("AnalyzeCvActivity", cv_url)
        if cv_content is None:
            logging.warning(f"Orchestrator ({instance_id}): AnalyzeCvActivity failed. Continuing with empty CV content.")
            cv_content = ""  # Use empty string instead of failing
        orchestration_input['cvContent'] = cv_content # Add content for subsequent steps

        # Step 3: Get Job Details
        logging.info(f"Orchestrator ({instance_id}): Calling GetJobDetailsActivity.")
        job_details = yield context.call_activity("GetJobDetailsActivity", job_id)
        if not job_details: raise Exception(f"GetJobDetailsActivity failed for jobId {job_id}.")

        # --- ADJUSTMENT START ---
        # Step 4: Perform Scoring (Calls the single activity wrapping the 3-stage logic)
        logging.info(f"Orchestrator ({instance_id}): Calling PerformScoringActivity.")
        scoring_input = {"candidate_data": orchestration_input, "job_details": job_details}
        score_result = yield context.call_activity("PerformScoringActivity", scoring_input)
        if score_result is None or "final_score" not in score_result or "final_reasoning" not in score_result:
             # Make error more specific
             logging.error(f"Orchestrator ({instance_id}): PerformScoringActivity failed or returned invalid data: {score_result}")
             raise Exception("PerformScoringActivity failed or returned invalid data.")
        final_score = score_result["final_score"]
        final_reasoning = score_result["final_reasoning"]
        logging.info(f"Orchestrator ({instance_id}): Scoring complete. Score: {final_score}")
        # --- ADJUSTMENT END ---

        # Step 4.5: Check if score meets minimum threshold
        import os
        score_threshold = float(os.getenv("AI_SCORE_THRESHOLD", "0.0"))
        if final_score < score_threshold:
            logging.info(f"Orchestrator ({instance_id}): Score {final_score} is below threshold {score_threshold}. Skipping indexing.")
            # If updating existing application, delete it from index
            if existing_application_id:
                logging.info(f"Orchestrator ({instance_id}): Deleting existing application {application_id} from index.")
                delete_result = yield context.call_activity("DeleteApplicationActivity", {"applicationId": application_id})
                if delete_result and delete_result.get("success"):
                    return {"status": "Completed", "applicationId": application_id, "score": final_score, "indexed": False, "operation": "deleted"}
                else:
                    return {"status": "Completed", "applicationId": application_id, "score": final_score, "indexed": False, "operation": "delete_failed"}
            else:
                return {"status": "Completed", "applicationId": application_id, "score": final_score, "indexed": False, "operation": "skipped"}

        # Step 5: Generate Embeddings
        logging.info(f"Orchestrator ({instance_id}): Calling GenerateEmbeddingsActivity.")
        embedding_input = {"candidate_data": orchestration_input}
        embeddings = yield context.call_activity("GenerateEmbeddingsActivity", embedding_input)
        if embeddings is None: raise Exception("GenerateEmbeddingsActivity failed.")

        # Step 6: Prepare and Index Document
        logging.info(f"Orchestrator ({instance_id}): Preparing final document.")
        # Extract jobTitle from job_details
        job_title = job_details.get("header", {}).get("jobTitle", "")
        
        # Encode CV URL to handle spaces
        encoded_cv_url = _encode_cv_url(cv_url)
        
        # Parse GPA values in education to handle '3.87/4.00' format
        education_data = orchestration_input.get("education", [])
        if education_data and isinstance(education_data, list):
            for edu in education_data:
                if isinstance(edu, dict) and 'gpa' in edu:
                    edu['gpa'] = _parse_gpa(edu['gpa'])
        
        application_document = {
            "applicationId": application_id, "candidateId": candidate_id, "jobId": job_id,
            "jobTitle": job_title,
            "name": orchestration_input.get("fullName"), "email": orchestration_input.get("email"),
            "phone": orchestration_input.get("phoneNumber"),
            "submissionDate": context.current_utc_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "birthDate": orchestration_input.get("birthDate"),
            "birthPlace": orchestration_input.get("birthPlace"),
            "personalWebsiteUrl": orchestration_input.get("personalWebsiteUrl"),
            "gender": orchestration_input.get("gender"),
            "interest": _prepare_field_for_index(orchestration_input.get("interest"), "interest"),
            "religion": _prepare_field_for_index(orchestration_input.get("religion"), "religion"),
            "medical": _prepare_field_for_index(orchestration_input.get("medical"), "medical"),
            "placement": _prepare_field_for_index(orchestration_input.get("placement"), "placement"),
            "expectedSalary": orchestration_input.get("expectedSalary"),
            "benefit": _prepare_field_for_index(orchestration_input.get("benefit"), "benefit"),
            "cvUrl": encoded_cv_url, "aiScore": final_score, "aiReasoning": final_reasoning,
            "location": {
                "address": orchestration_input.get("address"),
                "city": orchestration_input.get("city"),
                "province": orchestration_input.get("province"),
                "country": orchestration_input.get("country"),
                "postCode": orchestration_input.get("postCode"),
            },
            "education": education_data,
            "workExperience": embeddings.get("workExperienceWithIndustries", orchestration_input.get("workExperience", [])),
            "cvContent": cv_content, "profileSummary": embeddings.get("profileSummaryText", ""),
            "profileSummaryVector": embeddings.get("profileSummaryVector", []),
            "cvContentVector": embeddings.get("cvContentVector", []),
            "industries": embeddings.get("industries", []),
        }

        logging.info(f"Orchestrator ({instance_id}): Calling IndexApplicationActivity.")
        index_result = yield context.call_activity("IndexApplicationActivity", {"documents": [application_document]})
        if not index_result or not index_result.get("success"):
            raise Exception(f"IndexApplicationActivity failed. Details: {index_result.get('error')}")

        logging.info(f"Orchestrator ({instance_id}): Processing completed successfully.")
        return {"status": "Completed", "applicationId": application_id, "score": final_score, "operation": operation_type}

    except Exception as e:
        error_msg = f"Orchestrator ({instance_id}): Failed: {e}"
        logging.error(error_msg, exc_info=True)
        context.set_custom_status({"status": "Failed", "error": str(e)})
        return {"status": "Failed", "error": str(e)}


# 3. Activity Functions for Application Processing
@bp.activity_trigger(input_name="inputData")
def FindExistingApplicationActivity(inputData: Dict[str, str]) -> Optional[str]:
    """Activity: Checks AI Search if an application exists for candidateId and jobId."""
    candidate_id = inputData.get("candidateId")
    job_id = inputData.get("jobId")
    logging.info(f"Activity: Searching for existing application: Candidate={candidate_id}, Job={job_id}")
    # Reuse the helper function (ensure it doesn't log excessively here)
    return _find_existing_application(candidate_id, job_id)

@bp.activity_trigger(input_name="cvUrl")
def AnalyzeCvActivity(cvUrl: str) -> Optional[str]:
    """Activity: Analyzes CV content from a URL using Document Intelligence."""
    # Encode the URL to handle spaces in filenames
    encoded_cv_url = _encode_cv_url(cvUrl)
    logging.info(f"Activity: Analyzing CV from URL: {encoded_cv_url}")
    try:
        analysis_request = {"urlSource": encoded_cv_url}
        poller = services.document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            analysis_request
        )
        result = poller.result()
        logging.info("Activity: CV Analysis successful.")
        return result.content
    except Exception as e:
        logging.error(f"Activity Error (AnalyzeCvActivity): {e}", exc_info=True)
        return None # Return None on failure

# @bp.activity_trigger(input_name="inputData")
# def ExtractSubScoresActivity(inputData: Dict[str, Any]) -> Optional[Dict[str, float]]:
#     """Activity: Calls AI to extract sub-scores."""
#     candidate_data = inputData.get("candidate_data")
#     job_details_input = inputData.get("job_details") # Get raw input
#     logging.info("Activity: Extracting AI sub-scores.")

#     # --- ADD TYPE CHECK AND PARSING ---
#     job_details = None
#     if isinstance(job_details_input, dict):
#         job_details = job_details_input
#         logging.info("Activity: job_details received as dict.")
#     elif isinstance(job_details_input, str):
#         try:
#             job_details = json.loads(job_details_input)
#             logging.warning("Activity: job_details received as string, parsed back to dict.")
#         except json.JSONDecodeError as e:
#             logging.error(f"Activity Error: Failed to parse job_details string: {e}. Input: {job_details_input}")
#             return None # Fail activity if parsing fails
#     else:
#         logging.error(f"Activity Error: Unexpected type for job_details: {type(job_details_input)}")
#         return None # Fail activity on unexpected type
#     # --- END TYPE CHECK ---

#     # Ensure candidate_data is also a dict (less likely to be stringified, but good practice)
#     if not isinstance(candidate_data, dict):
#          logging.error(f"Activity Error: Unexpected type for candidate_data: {type(candidate_data)}")
#          return None


#     # Use the internal function from ai_components (ensure it's imported correctly)
#     try:
#         from helper_functions.component import _extract_ai_sub_scores
#         sub_scores = _extract_ai_sub_scores(candidate_data, job_details) # Pass the potentially parsed job_details
#         if sub_scores:
#             logging.info("Activity: AI Sub-scores extracted successfully.")
#         else:
#             logging.error("Activity Error (ExtractSubScoresActivity): _extract_ai_sub_scores returned None.")
#         return sub_scores
#     except ImportError:
#          logging.error("Activity Error: Could not import _extract_ai_sub_scores.")
#          return None
#     except Exception as e:
#          logging.error(f"Activity Error during sub-score extraction call: {e}", exc_info=True)
#          return None

# @bp.activity_trigger(input_name="inputData")
# def CalculateScoreActivity(inputData: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#     """Activity: Calculates final score and reasoning in Python."""
#     ai_sub_scores = inputData.get("ai_sub_scores")
#     job_details_input = inputData.get("job_details")
#     logging.info("Activity: Calculating final score and reasoning.")

#     # --- ADD TYPE CHECK AND PARSING ---
#     job_details = None
#     if isinstance(job_details_input, dict):
#         job_details = job_details_input
#     elif isinstance(job_details_input, str):
#         try:
#             job_details = json.loads(job_details_input)
#             logging.warning("Activity: job_details received as string in CalculateScore, parsed back.")
#         except json.JSONDecodeError as e:
#             logging.error(f"Activity Error: Failed to parse job_details string in CalculateScore: {e}")
#             return None
#     else:
#         logging.error(f"Activity Error: Unexpected type for job_details in CalculateScore: {type(job_details_input)}")
#         return None
#     # Ensure ai_sub_scores is a dict
#     if not isinstance(ai_sub_scores, dict):
#          logging.error(f"Activity Error: Unexpected type for ai_sub_scores: {type(ai_sub_scores)}")
#          return None
#     # --- END TYPE CHECK ---

#     try:
#         # Use the internal function from ai_components
#         from helper_functions.component import _calculate_score_and_reasoning
#         final_score, final_reasoning = _calculate_score_and_reasoning(ai_sub_scores, job_details)
#         logging.info(f"Activity: Final score calculated: {final_score}")
#         return {"final_score": final_score, "final_reasoning": final_reasoning}
#     except ImportError:
#          logging.error("Activity Error: Could not import _calculate_score_and_reasoning.")
#          return None
#     except Exception as e:
#         logging.error(f"Activity Error (CalculateScoreActivity): {e}", exc_info=True)
#         return None

@bp.activity_trigger(input_name="inputData")
def PerformScoringActivity(inputData: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Activity: Performs the complete 3-stage scoring process by calling get_ai_score."""
    candidate_data = inputData.get("candidate_data")
    job_details_input = inputData.get("job_details")
    logging.info("Activity: Performing full scoring process.")

    # --- Add Type Check for job_details ---
    job_details = None
    if isinstance(job_details_input, dict):
        job_details = job_details_input
    elif isinstance(job_details_input, str):
        try:
            job_details = json.loads(job_details_input)
            logging.warning("Activity: job_details (scoring) received as string, parsed.")
        except json.JSONDecodeError as e:
            logging.error(f"Activity Error: Failed to parse job_details string in PerformScoring: {e}")
            return None
    else:
        logging.error(f"Activity Error: Unexpected type for job_details in PerformScoring: {type(job_details_input)}")
        return None
    if not isinstance(candidate_data, dict):
        logging.error(f"Activity Error: Unexpected type for candidate_data in PerformScoring: {type(candidate_data)}")
        return None
    # --- End Type Check ---

    try:
        # Call the main get_ai_score function which handles the 3 stages
        logging.info(f"Activity: Calling get_ai_score for candidate {candidate_data.get('candidateId')} and job {job_details.get('header', {}).get('jobId', 'unknown')}")
        final_score, final_reasoning = get_ai_score(candidate_data, job_details)

        # Basic check on results
        if isinstance(final_score, float) and isinstance(final_reasoning, str):
            logging.info(f"Activity: Scoring complete. Score: {final_score}")
            return {"final_score": final_score, "final_reasoning": final_reasoning}
        else:
            logging.error(f"Activity Error: get_ai_score returned invalid types. Score: {type(final_score)} = {final_score}, Reasoning: {type(final_reasoning)} = {final_reasoning}")
            return None # Indicate failure

    except Exception as e:
        # Catch errors from within get_ai_score if they weren't handled internally
        logging.error(f"Activity Error (PerformScoringActivity): Exception during get_ai_score call for candidate {candidate_data.get('candidateId', 'unknown')}: {e}", exc_info=True)
        return None # Indicate failure

@bp.activity_trigger(input_name="inputData")
def GenerateEmbeddingsActivity(inputData: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Activity: Generates profile summary text, embeddings, and extracts industries."""
    candidate_data = inputData.get("candidate_data")
    logging.info("Activity: Generating embeddings and extracting industries.")
    try:
        # Generate summary text first
        profile_summary = format_candidate_details_for_prompt(candidate_data)
        if not profile_summary:
             logging.warning("Activity: Profile summary text is empty.")
             # Return empty structure? Or fail? Returning empty for now.
             return {"profileSummaryText": "", "profileSummaryVector": [], "cvContentVector": [], "industries": [], "workExperienceWithIndustries": []}


        # Import functions
        from helper_functions.component import generate_embedding, extract_candidate_industries

        profile_vector = generate_embedding(profile_summary)
        cv_vector = generate_embedding(candidate_data.get('cvContent'))
        
        # Extract industries from work experience
        work_experience = candidate_data.get('workExperience', [])
        per_job_industries, unique_industries = extract_candidate_industries(work_experience)
        
        # Merge industries back into work experience
        work_experience_with_industries = []
        for i, exp in enumerate(work_experience):
            exp_with_industry = exp.copy()  # Don't modify original
            exp_with_industry['industry'] = per_job_industries[i] if i < len(per_job_industries) else "Other"
            work_experience_with_industries.append(exp_with_industry)

        logging.info(f"Activity: Embeddings and industries generated successfully. Unique industries: {unique_industries}")
        return {
            "profileSummaryText": profile_summary,
            "profileSummaryVector": profile_vector if profile_vector else [],
            "cvContentVector": cv_vector if cv_vector else [],
            "industries": unique_industries,  # Flat unique list
            "workExperienceWithIndustries": work_experience_with_industries  # Work experience with industry field
        }
    except Exception as e:
        logging.error(f"Activity Error (GenerateEmbeddingsActivity): {e}", exc_info=True)
        return None


@bp.activity_trigger(input_name="inputData")
def DeleteApplicationActivity(inputData: Dict[str, Any]) -> Dict[str, Any]:
    """Activity: Deletes an application document from AI Search."""
    application_id = inputData.get("applicationId")
    if not application_id:
        logging.warning("Activity: No applicationId provided for deletion.")
        return {"success": False, "error": "No applicationId provided"}
    
    logging.info(f"Activity: Deleting application {application_id} from index.")
    try:
        result = services.search_client.delete_documents(documents=[{"applicationId": application_id}])
        if result and len(result) > 0 and result[0].succeeded:
            logging.info(f"Activity: Successfully deleted application {application_id}.")
            return {"success": True}
        else:
            error_msg = result[0].error_message if result and len(result) > 0 else "Unknown error"
            logging.error(f"Activity: Failed to delete application {application_id}. Error: {error_msg}")
            return {"success": False, "error": error_msg}
    except Exception as e:
        logging.error(f"Activity Error (DeleteApplicationActivity): {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@bp.activity_trigger(input_name="inputData")
def IndexApplicationActivity(inputData: Dict[str, Any]) -> Dict[str, Any]:
    """Activity: Indexes one or more application documents into AI Search."""
    documents: List[Dict[str, Any]] = inputData.get("documents", [])
    if not documents:
        logging.warning("Activity: No documents provided for indexing.")
        return {"success": True, "processed": 0} # No-op is successful

    logging.info(f"Activity: Indexing {len(documents)} document(s).")
    
    # Debug: Log field types to identify array issues
    if documents:
        doc = documents[0]
        for key, value in doc.items():
            if isinstance(value, list) and value and not isinstance(value[0], (dict, float, int)):
                logging.warning(f"DEBUG: Field '{key}' is a simple list (might need conversion): {value[:2]}...")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                logging.info(f"DEBUG: Field '{key}' is complex list (should be OK): {len(value)} items")
            elif isinstance(value, list):
                logging.info(f"DEBUG: Field '{key}' is list: {type(value[0]) if value else 'empty'}, length: {len(value)}")
    
    try:
        # Use merge_or_upload_documents for create/update
        results = services.search_client.merge_or_upload_documents(documents=documents)
        successful_count = sum(1 for r in results if r.succeeded)
        failed_count = len(results) - successful_count
        logging.info(f"Activity: Indexing complete. Succeeded: {successful_count}, Failed: {failed_count}")

        if failed_count > 0:
            error_details = []
            for i, result in enumerate(results):
                if not result.succeeded:
                    app_id = documents[i].get('applicationId', 'UnknownID')
                    error_details.append(f"AppID {app_id}: {result.error_message}")
            error_msg = f"Failed to index {failed_count} documents. Errors: {'; '.join(error_details)}"
            logging.error(error_msg)
            # Return failure status but include success count
            return {"success": False, "processed": successful_count, "failed": failed_count, "error": error_msg}
        else:
            return {"success": True, "processed": successful_count, "failed": 0}

    except Exception as e:
        logging.error(f"Activity Error (IndexApplicationActivity): {e}", exc_info=True)
        return {"success": False, "processed": 0, "failed": len(documents), "error": str(e)}

# --- Profile Update Workflow ---

# 1. HTTP Starter for Update Profile
@bp.route(route="update_profile")
@bp.durable_client_input(client_name="client")
async def http_update_profile_starter(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    """
    HTTP endpoint to start the profile update and re-scoring orchestration.
    """
    logging.info('HTTP Starter: update_profile request received.')
    try:
        updated_candidate_data = req.get_json()
        candidate_id = updated_candidate_data.get('candidateId')
        jobs_applied_list = updated_candidate_data.get('jobsApplied')
        new_cv_url = updated_candidate_data.get('cv')

        
        if (not candidate_id or candidate_id.strip() == "" or 
            jobs_applied_list is None or not isinstance(jobs_applied_list, list) or 
            not new_cv_url or new_cv_url.strip() == ""):
            logging.warning("HTTP Starter: Missing or invalid fields in update_profile request.")
            return func.HttpResponse("Missing or invalid fields: candidateId (string), jobsApplied (list), cv (URL)", status_code=400)

        instance_id = await client.start_new(
            orchestration_function_name="ProfileUpdateOrchestrator",
            client_input=updated_candidate_data
        )
        logging.info(f"HTTP Starter: Started profile update orchestration with ID = '{instance_id}'.")
        return client.create_check_status_response(req, instance_id)

    except ValueError:
        return func.HttpResponse("Invalid JSON format in request body.", status_code=400)
    except Exception as e:
        logging.error(f"HTTP Starter Error (update_profile): {e}", exc_info=True)
        return func.HttpResponse("Internal server error.", status_code=500)

# 2. Orchestrator for Profile Update (Fan-Out/Fan-In)
@bp.orchestration_trigger(context_name="context")
def ProfileUpdateOrchestrator(context: df.DurableOrchestrationContext):
    """
    Orchestrates updating a candidate's profile and re-scoring multiple applications.
    Uses Fan-Out/Fan-In pattern.
    """
    updated_candidate_data = context.get_input()
    instance_id = context.instance_id
    logging.info(f"Orchestrator ({instance_id}): Starting profile update process.")

    candidate_id = updated_candidate_data.get('candidateId')
    jobs_applied_list = updated_candidate_data.get('jobsApplied', [])
    new_cv_url = updated_candidate_data.get('cv')

    # Basic check
    if not all([candidate_id, new_cv_url]): # jobs_applied_list can be empty, handled later
         error_msg = f"Orchestrator ({instance_id}): Invalid input received for profile update."
         logging.error(error_msg)
         return {"status": "Failed", "error": error_msg}

    try:
        # Step 1: Analyze NEW CV content (once)
        logging.info(f"Orchestrator ({instance_id}): Analyzing updated CV.")
        new_cv_content = yield context.call_activity("AnalyzeCvActivity", new_cv_url)
        if new_cv_content is None:
            logging.warning(f"Orchestrator ({instance_id}): AnalyzeCvActivity failed for updated CV. Continuing with empty CV content.")
            new_cv_content = ""  # Use empty string instead of failing
        updated_candidate_data['cvContent'] = new_cv_content

        # Step 2: Generate NEW Embeddings (once)
        logging.info(f"Orchestrator ({instance_id}): Generating new embeddings.")
        embedding_input = {"candidate_data": updated_candidate_data}
        new_embeddings = yield context.call_activity("GenerateEmbeddingsActivity", embedding_input)
        if new_embeddings is None:
            raise Exception("GenerateEmbeddingsActivity failed for updated profile.")

        # Step 3: Find existing applications to update
        logging.info(f"Orchestrator ({instance_id}): Finding existing applications.")
        if not jobs_applied_list:
             logging.warning(f"Orchestrator ({instance_id}): No job IDs provided, nothing to update.")
             return {"status": "Completed", "updated_count": 0, "updated_scores": []}

        # Call an activity to perform the search
        search_input = {"candidateId": candidate_id, "jobIds": jobs_applied_list}
        original_app_details = yield context.call_activity("FindApplicationsForUpdateActivity", search_input)

        if not original_app_details:
             logging.warning(f"Orchestrator ({instance_id}): No existing applications found matching job IDs.")
             return {"status": "Completed", "updated_count": 0, "updated_scores": []}

        logging.info(f"Orchestrator ({instance_id}): Found {len(original_app_details)} applications to re-score.")

        # Step 4: Fan-Out - Create parallel tasks to re-score each application
        rescore_tasks = []
        for app_detail in original_app_details:
            # Each task will call a sub-orchestrator or a sequence of activities
            # Using call_sub_orchestrator is generally more robust for multi-step tasks
            # For simplicity here, let's call an activity that does the re-score sequence
            rescore_input = {
                "application_id": app_detail.get('applicationId'),
                "job_id": app_detail.get('jobId'),
                "updated_candidate_data": updated_candidate_data, # Includes new cvContent
                "new_embeddings": new_embeddings
            }
            rescore_tasks.append(context.call_activity("RescoreApplicationActivity", rescore_input))

        # Step 5: Fan-In - Wait for all re-scoring tasks to complete
        logging.info(f"Orchestrator ({instance_id}): Waiting for {len(rescore_tasks)} re-score tasks.")
        updated_documents = yield context.task_all(rescore_tasks)
        logging.info(f"Orchestrator ({instance_id}): All re-score tasks completed.")

        # Extract scores from all documents for reporting (including below-threshold)
        updated_scores = []
        for doc in updated_documents:
            if doc:
                if doc.get("_delete"):
                    # Include deleted applications with their scores
                    updated_scores.append({
                        "applicationId": doc.get("applicationId"),
                        "jobId": doc.get("jobId"),
                        "aiScore": doc.get("aiScore"),
                        "belowThreshold": True
                    })
                else:
                    # Include updated applications with their scores
                    updated_scores.append({
                        "applicationId": doc.get("applicationId"),
                        "jobId": doc.get("jobId"),
                        "aiScore": doc.get("aiScore"),
                        "belowThreshold": False
                    })

        # Separate documents for deletion and update based on threshold
        documents_to_delete = [doc for doc in updated_documents if doc and doc.get("_delete")]
        documents_to_update = [doc for doc in updated_documents if doc and not doc.get("_delete")]
        failed_rescore_count = len(original_app_details) - len([doc for doc in updated_documents if doc])
        
        deleted_count = 0
        # Step 6a: Delete below-threshold applications
        if documents_to_delete:
            logging.info(f"Orchestrator ({instance_id}): Deleting {len(documents_to_delete)} below-threshold applications.")
            delete_tasks = []
            for doc in documents_to_delete:
                delete_tasks.append(context.call_activity("DeleteApplicationActivity", {"applicationId": doc["applicationId"]}))
            delete_results = yield context.task_all(delete_tasks)
            deleted_count = sum(1 for r in delete_results if r and r.get("success"))
            logging.info(f"Orchestrator ({instance_id}): Deleted {deleted_count} applications.")

        # Step 6b: Bulk Index the updated documents
        if documents_to_update:
            logging.info(f"Orchestrator ({instance_id}): Indexing {len(documents_to_update)} updated documents.")
            index_result = yield context.call_activity("IndexApplicationActivity", {"documents": documents_to_update})
            if not index_result or not index_result.get("success"):
                 # Log error but maybe still report partial success?
                 logging.error(f"Orchestrator ({instance_id}): Indexing failed for some documents. Details: {index_result.get('error')}")
                 # Decide on final status based on indexing result
                 final_status = {"status": "PartiallyCompleted", "updated_count": index_result.get("processed", 0), "deleted_count": deleted_count, "rescore_failures": failed_rescore_count, "index_failures": index_result.get("failed", len(documents_to_update)), "updated_scores": updated_scores}
            else:
                 final_status = {"status": "Completed", "updated_count": len(documents_to_update), "deleted_count": deleted_count, "rescore_failures": failed_rescore_count, "updated_scores": updated_scores}
        else:
             logging.warning(f"Orchestrator ({instance_id}): No documents to update.")
             final_status = {"status": "Completed", "updated_count": 0, "deleted_count": deleted_count, "rescore_failures": failed_rescore_count, "updated_scores": updated_scores}

        logging.info(f"Orchestrator ({instance_id}): Profile update orchestration finished.")
        context.set_custom_status(final_status)
        return final_status

    except Exception as e:
        error_msg = f"Orchestrator ({instance_id}): Profile update failed with error: {e}"
        logging.error(error_msg, exc_info=True)
        context.set_custom_status({"status": "Failed", "error": str(e)})
        return {"status": "Failed", "error": str(e)}

# 3. Activity Functions for Profile Update
@bp.activity_trigger(input_name="inputData")
def FindApplicationsForUpdateActivity(inputData: Dict[str, Any]) -> List[Dict[str, str]]:
    """Activity: Finds existing applicationIds and jobIds for a candidate and list of jobs."""
    candidate_id = inputData.get("candidateId")
    job_ids = inputData.get("jobIds", [])
    logging.info(f"Activity: Finding applications for candidate {candidate_id} in jobs: {job_ids}")

    if not job_ids: return []

    # Construct filter efficiently
    jobs_filter_part = " or ".join([f"jobId eq '{job_id}'" for job_id in job_ids])
    search_filter = f"candidateId eq '{candidate_id}' and ({jobs_filter_part})"

    try:
        results = services.search_client.search(
            search_text="", filter=search_filter, select="applicationId, jobId"
        )
        apps_found = [{"applicationId": r.get('applicationId'), "jobId": r.get('jobId')} for r in results if r.get('applicationId') and r.get('jobId')]
        logging.info(f"Activity: Found {len(apps_found)} applications matching criteria.")
        return apps_found
    except Exception as e:
        logging.error(f"Activity Error (FindApplicationsForUpdateActivity): {e}", exc_info=True)
        return [] # Return empty list on error

# Activity Function for Re-scoring (ADJUSTED)
@bp.activity_trigger(input_name="inputData")
def RescoreApplicationActivity(inputData: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Activity: Re-scores a single application using updated candidate data by calling get_ai_score."""
    application_id = inputData.get("application_id")
    job_id = inputData.get("job_id")
    updated_candidate_data = inputData.get("updated_candidate_data")
    new_embeddings = inputData.get("new_embeddings", {})
    logging.info(f"Activity: Re-scoring application {application_id} for job {job_id}")

    # --- Type Checks ---
    if not isinstance(updated_candidate_data, dict):
        logging.error(f"Activity Error (Rescore): Invalid type for updated_candidate_data: {type(updated_candidate_data)}")
        return None
    if not isinstance(new_embeddings, dict):
         logging.error(f"Activity Error (Rescore): Invalid type for new_embeddings: {type(new_embeddings)}")
         return None
    # --- End Type Checks ---

    try:
        # 1. Get ORIGINAL Job Details
        from helper_functions.job_api_client import get_job_details as get_job_details_async
        import asyncio
        original_job_details = asyncio.run(get_job_details_async(job_id))
        if not original_job_details:
            raise Exception(f"Original job details not found for {job_id}")

        # --- ADJUSTMENT: Call the main get_ai_score function ---
        # This function now handles the 3-stage process internally
        new_score, new_reasoning = get_ai_score(updated_candidate_data, original_job_details)
        # --- END ADJUSTMENT ---

        # Basic check on results from get_ai_score
        if not isinstance(new_score, float) or not isinstance(new_reasoning, str) or "Gagal" in new_reasoning or "Error" in new_reasoning:
            # Handle potential failure from get_ai_score
            logging.error(f"Activity Error (Rescore): get_ai_score failed or returned invalid data for AppID {application_id}. Score: {new_score}, Reason: {new_reasoning}")
            # Do not return a document fragment if scoring failed
            return None

        # Check if new score meets minimum threshold
        import os
        score_threshold = float(os.getenv("AI_SCORE_THRESHOLD", "0.0"))
        if new_score < score_threshold:
            logging.info(f"Activity (Rescore): New score {new_score} is below threshold {score_threshold} for AppID {application_id}. Marking for deletion.")
            # Return a special marker document to signal deletion with score info
            return {
                "applicationId": application_id,
                "jobId": job_id,
                "aiScore": new_score,
                "_delete": True  # Special marker for deletion
            }

        # 4. Prepare the updated document fragment
        # Extract jobTitle from job_details
        job_title = original_job_details.get("header", {}).get("jobTitle", "")
        
        # Encode CV URL to handle spaces
        cv_url_raw = updated_candidate_data.get("cv")
        encoded_cv_url = _encode_cv_url(cv_url_raw)
        
        # Parse GPA values in education to handle '3.87/4.00' format
        education_data = updated_candidate_data.get("education", [])
        if education_data and isinstance(education_data, list):
            for edu in education_data:
                if isinstance(edu, dict) and 'gpa' in edu:
                    edu['gpa'] = _parse_gpa(edu['gpa'])
        
        updated_doc_fragment = {
            "applicationId": application_id,
            "candidateId": updated_candidate_data.get('candidateId'),
            "jobId": job_id,
            "jobTitle": job_title,
            "name": updated_candidate_data.get("fullName"),
            "email": updated_candidate_data.get("email"),
            "phone": updated_candidate_data.get("phoneNumber"),
            "location": {
                "address": updated_candidate_data.get("address"),
                "city": updated_candidate_data.get("city"),
                "province": updated_candidate_data.get("province"),
                "country": updated_candidate_data.get("country"),
                "postCode": updated_candidate_data.get("postCode"),
            },
            "education": education_data,
            "workExperience": new_embeddings.get("workExperienceWithIndustries", updated_candidate_data.get("workExperience", [])),
            "cvUrl": encoded_cv_url,
            "cvContent": updated_candidate_data.get('cvContent'),
            "birthDate": updated_candidate_data.get("birthDate"),
            "birthPlace": updated_candidate_data.get("birthPlace"),
            "personalWebsiteUrl": updated_candidate_data.get("personalWebsiteUrl"),
            "gender": updated_candidate_data.get("gender"),
            "interest": _prepare_field_for_index(updated_candidate_data.get("interest"), "interest"),
            "religion": _prepare_field_for_index(updated_candidate_data.get("religion"), "religion"),
            "medical": _prepare_field_for_index(updated_candidate_data.get("medical"), "medical"),
            "placement": _prepare_field_for_index(updated_candidate_data.get("placement"), "placement"),
            "expectedSalary": updated_candidate_data.get("expectedSalary"),
            "benefit": _prepare_field_for_index(updated_candidate_data.get("benefit"), "benefit"),
            "profileSummary": new_embeddings.get("profileSummaryText", ""),
            "profileSummaryVector": new_embeddings.get("profileSummaryVector", []),
            "cvContentVector": new_embeddings.get("cvContentVector", []),
            "industries": new_embeddings.get("industries", []),
            "aiScore": new_score,
            "aiReasoning": new_reasoning,
        }
        logging.info(f"Activity: Successfully re-scored application {application_id}. New score: {new_score}")
        return updated_doc_fragment

    except Exception as e:
        logging.error(f"Activity Error (RescoreApplicationActivity) for AppID {application_id}: {e}", exc_info=True)
        return None # Signal failure


# --- ranked-candidates (Standard HTTP Trigger - Remains Unchanged) ---
# It reads the latest data from AI Search, which is updated by the orchestrations.
@bp.route(route="ranked-candidates")
def GetRankedCandidates(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP Trigger to retrieve ranked applications for a specific job.
    Reads the latest scores from AI Search.
    """
    # ... (Keep the exact same code as the previous version) ...
    logging.info('GetRankedCandidates function processed a request.')
    job_id = req.params.get('jobId')
    user_id = req.params.get('userId')
    top_n = int(req.params.get('top', 5))

    if not all([job_id, user_id]):
        return func.HttpResponse("Missing required parameters: jobId, userId", status_code=400)

    try:
        logging.info(f"Querying AI Search for top {top_n} applications for job ID {job_id}.")
        results = services.search_client.search(
            search_text="*", filter=f"jobId eq '{job_id}'",
            order_by="aiScore desc", top=top_n,
            select="applicationId,candidateId,name,aiScore,aiReasoning,education,workExperience,location,cvContent"
        )

        table_rows = []
        result_details = []
        from helper_functions.job_api_client import get_job_details as get_job_details_async
        import asyncio
        job_details = asyncio.run(get_job_details_async(job_id))
        if not job_details:
            job_details = {"header": {"jobTitle": "the relevant position"}, "items": []}
        job_title = job_details.get("header", {}).get("jobTitle", "the relevant position")
        job_location_item = next((item for item in job_details.get("items", []) if item.get("categoryName") == "Work Location"), None)
        job_location = job_location_item.get("value", "the relevant area") if job_location_item else "the relevant area"

        application_list = list(results)
        if not application_list:
             logging.warning(f"No applications found for jobId {job_id} in AI Search.")
             final_response = {
                 "jobId": job_id, "userId": user_id,
                 "response": f"Tidak ada kandidat (aplikasi) yang ditemukan untuk posisi {job_title} di area {job_location}.",
                 "responseTable": {"columns": [], "rows": []}, "result": []
             }
             return func.HttpResponse(json.dumps(final_response, ensure_ascii=False), status_code=200, mimetype="application/json")

        logging.info("Summarizing applicant profiles for the response table.")
        from helper_functions.component import summarize_candidate_for_table # Import needed function
        for application in application_list:
            candidate_location_city = application.get("location", {}).get("city") if isinstance(application.get("location"), dict) else None
            summary_input = {
                "name": application.get("name"), "location": {"city": candidate_location_city} if candidate_location_city else None,
                "workExperience": application.get("workExperience"), "education": application.get("education"), "cvContent": application.get("cvContent")
            }
            summary = summarize_candidate_for_table(summary_input)
            table_rows.append({
                "name": application.get("name", "N/A"), "experience": summary.get("experience", "N/A"), "skill": summary.get("skill", "N/A"),
                "location": summary.get("location", "N/A"), "lastWork": summary.get("lastWork", "N/A"), "education": summary.get("education", "N/A"),
            })
            result_details.append({
                "applicantId": application.get("candidateId", "N/A"), "applicantName": application.get("name", "N/A"),
                "score": application.get("aiScore", 0.0), "reason": application.get("aiReasoning", "N/A")
            })

        final_response = {
            "jobId": job_id, "userId": user_id,
            "response": f"Berikut beberapa kandidat (aplikasi) yang sesuai dengan kriteria Anda untuk posisi {job_title} di area {job_location}:",
            "responseTable": {
                "columns": [ { "field": "name", "label": "Nama" }, { "field": "experience", "label": "Pengalaman" }, { "field": "skill", "label": "Keahlian" }, { "field": "location", "label": "Lokasi" }, { "field": "lastWork", "label": "Terakhir Bekerja di" }, { "field": "education", "label": "Pendidikan" } ],
                "rows": table_rows
            },
            "result": result_details
        }
        return func.HttpResponse(json.dumps(final_response, ensure_ascii=False), status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(f"An error occurred while getting ranked candidates: {e}", exc_info=True)
        return func.HttpResponse("Internal Server Error.", status_code=500)


# --- Register the Blueprint with the main Function App ---
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
app.register_functions(bp) # Register all functions defined in the blueprint