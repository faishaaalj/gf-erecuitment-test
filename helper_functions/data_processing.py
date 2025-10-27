from bs4 import BeautifulSoup
import re # Import regex for safer key generation
import logging

logger = logging.getLogger(__name__)

def _create_safe_dict_key(text: str) -> str:
    """Converts a string into a safer dictionary key format for JSON keys used by AI."""
    if not text or not isinstance(text, str):
        logger.warning(f"Invalid input to _create_safe_dict_key: {text}. Using 'Unknown'.")
        return "Unknown"
    # Convert to lowercase first? Optional, but helps consistency.
    text = text.lower()
    # Replace spaces and common problematic characters with underscores
    safe_text = re.sub(r'[()\s/.,;:%]+', '_', text) # Added %
    # Remove any non-alphanumeric characters (except underscore)
    safe_text = re.sub(r'[^a-zA-Z0-9_]', '', safe_text)
    # Remove any leading/trailing/multiple underscores
    safe_text = re.sub(r'_+', '_', safe_text).strip('_')
    # Handle potential empty string after stripping
    if not safe_text:
        # Try a simpler replacement if the above fails
        safe_text = re.sub(r'\W+', '_', text).strip('_')
        if not safe_text:
            logger.warning(f"Resulting safe key is empty for input: {text}. Using 'EmptyValue'.")
            return "EmptyValue"
    # Optional: Truncate long keys if necessary
    # MAX_KEY_LEN = 50
    # if len(safe_text) > MAX_KEY_LEN:
    #     safe_text = safe_text[:MAX_KEY_LEN]
    return safe_text


def clean_html(raw_html: str) -> str:
    """
    Cleans any HTML content from a string and formats it into readable plain text.
    - Preserves list structures by adding bullet points to <li> tags.
    - Converts block-level tags like <p>, <div>, and <br> into newlines.
    - Returns plain text if the input is already plain text.
    """
    if not raw_html or not isinstance(raw_html, str) or not raw_html.strip():
        return "Not specified."

    # Parse the HTML content using html.parser for robustness
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
    except Exception as e:
        logger.error(f"Error parsing HTML in clean_html: {e}. Returning raw input.")
        return raw_html # Return original if parsing fails

    # Check if the content was actually HTML. If no tags found, return original.
    # Check specifically for tags other than potential simple formatting like <b> etc.
    # A simple check for '<' and '>' might be sufficient and faster if needed.
    if not soup.find(True, recursive=False): # Check for any top-level tags
         # Check more thoroughly for common block tags if the simple check fails
         if not soup.find(['p', 'div', 'ul', 'ol', 'li', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
             logger.debug("clean_html received plain text, returning as is.")
             return raw_html.strip() # Strip whitespace just in case

    # Handle list items - modify soup in place
    for li in soup.find_all('li'):
        # Check if bullet point already exists (e.g., "- item")
        if not li.get_text(strip=True).startswith('-'):
             li.insert(0, '- ')

    # Extract text with newlines between block elements
    text = soup.get_text(separator='\n', strip=True)

    return text

def format_job_details_for_prompt(job_details: dict) -> str:
    """
    Formats the structured job details JSON into a readable string for the AI prompt,
    including weighted score contributions and mandatory tags. Used for context in AI prompts.
    """
    details = []
    header = job_details.get("header", {})
    details.append(f"Job Title: {header.get('jobTitle', 'N/A')} (ID: {header.get('jobId', 'N/A')})")
    details.append("\n--- EVALUATION CRITERIA ---")

    # Format Requirements
    requirements_data = job_details.get('requirements', {})
    if isinstance(requirements_data, dict):
        req_desc = requirements_data.get('description', 'Not specified.')
        req_score = requirements_data.get('scoreContribution', 0)
    else:
        req_desc = requirements_data if requirements_data else 'Not specified.'
        req_score = job_details.get('scoreContributionRequirements', 0)
    req_text = clean_html(req_desc) # Clean the description
    req_mandatory_tag = " (MANDATORY)" if req_score == 100 else ""
    # Only add if weight > 0 or it's mandatory (weight 100)
    if req_score > 0:
        details.append(f"\n1. General Requirements (Weight: {int(req_score)}/100){req_mandatory_tag}:\n{req_text}")

    # Format Responsibilities
    responsibilities_data = job_details.get('responsibilities', {})
    if isinstance(responsibilities_data, dict):
        resp_desc = responsibilities_data.get('description', 'Not specified.')
        resp_score = responsibilities_data.get('scoreContribution', 0)
    else:
        resp_desc = responsibilities_data if responsibilities_data else 'Not specified.'
        resp_score = job_details.get('scoreContributionResponsibilities', 0)
    resp_text = clean_html(resp_desc) # Clean the description
    resp_mandatory_tag = " (MANDATORY)" if resp_score == 100 else ""
    # Only add if weight > 0 or it's mandatory (weight 100)
    if resp_score > 0:
        details.append(f"\n2. Key Responsibilities (Weight: {int(resp_score)}/100){resp_mandatory_tag}:\n{resp_text}")

    # Format Specific Item Criteria
    items = job_details.get("items", []) or [] # Ensure items is a list
    if items:
        details.append("\n3. Specific Criteria:")
        item_criteria_added = False
        for i, item in enumerate(items):
            score = item.get("scoreContribution", 0)
            # Only include criteria with weight > 0 in the prompt context
            if score > 0:
                item_criteria_added = True
                name = item.get("categoryName", f"Item_{i+1}")
                value = item.get("value", "N/A")
                mandatory_tag = " (MANDATORY)" if score == 100 else ""
                details.append(f"  - {name}: Target '{value}' (Weight: {int(score)}/100){mandatory_tag}")
        if not item_criteria_added:
             details.append("  (No specific item criteria with weight > 0)")


    return "\n".join(details)

def format_candidate_details_for_prompt(candidate_data: dict) -> str:
    """
    Formats the structured candidate data (including CV content)
    into a readable string for use in AI prompts for scoring.
    """
    details = []
    logger.debug(f"Formatting candidate data for prompt: {candidate_data.get('candidateId', 'N/A')}")


    details.append(f"- Candidate ID: {candidate_data.get('candidateId', 'N/A')}")
    details.append(f"- Name: {candidate_data.get('fullName', 'N/A')}") # Use 'fullName'
    details.append(f"- Email: {candidate_data.get('email', 'N/A')}")
    details.append(f"- Phone: {candidate_data.get('phoneNumber', 'N/A')}") # Use 'phoneNumber'
    details.append(f"- Location: {candidate_data.get('city', 'N/A')}")
    details.append(f"- Address: {candidate_data.get('address', 'N/A')}")
    # Add other potentially relevant fields if available in candidate_data
    details.append(f"- Gender: {candidate_data.get('gender', 'N/A')}")
    details.append(f"- Birth Date: {candidate_data.get('birthDate', 'N/A')}")


    # Format Education History
    education_list = candidate_data.get('education', []) or []
    if education_list:
        details.append("\nEducation History:")
        for edu in education_list:
            details.append(f"  - School: {edu.get('schoolName', 'N/A')}, Major: {edu.get('major', 'N/A')}, Degree: {edu.get('degree', 'N/A')}, GPA: {edu.get('gpa', 'N/A')}, Dates: {edu.get('dateFrom','')} to {edu.get('dateTo','')}")
    else:
        details.append("\nEducation History: Not Provided.")

    # Format Work Experience
    work_list = candidate_data.get('workExperience', []) or []
    if work_list:
        details.append("\nWork Experience:")
        for work in work_list:
            details.append(f"  - Title: {work.get('jobTitle', 'N/A')} at {work.get('companyName', 'N/A')} ({work.get('dateFrom','')} to {work.get('dateTo','')}), Level: {work.get('jobLevel', 'N/A')}, Function: {work.get('jobFunction', 'N/A')}")
    else:
         details.append("\nWork Experience: Not Provided.")

    # Add full CV content for comprehensive analysis
    cv_content = candidate_data.get('cvContent')
    details.append("\n--- Full CV Content ---")
    if cv_content and isinstance(cv_content, str) and cv_content.strip():
        details.append(cv_content)
    else:
        logger.warning(f"CV Content missing or empty for candidate {candidate_data.get('candidateId', 'N/A')}")
        details.append("CV content not provided or extracted.")

    formatted_string = "\n".join(details)
    logger.debug(f"Formatted candidate details for prompt (first 200 chars): {formatted_string[:200]}")
    return formatted_string

