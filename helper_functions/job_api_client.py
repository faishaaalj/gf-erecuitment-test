# filename: function-app-project/helper_functions/job_api_client.py
# (This new file manages login and fetching job details)

import requests
import os
import json
import logging
import datetime
from typing import Optional, Dict, Any
from azure.core.exceptions import ResourceNotFoundError

from .services import token_cache_container_client

logger = logging.getLogger(__name__)

JOB_API_BASE_URL = os.environ.get("JOB_API_BASE_URL", "")
JOB_API_USER = os.environ.get("JOB_API_USER", "")
JOB_API_PASSWORD = os.environ.get("JOB_API_PASSWORD", "")
TOKEN_CACHE_BLOB_NAME = "system/job_api_auth_token.json"


async def _get_cached_token() -> Optional[str]:
    """
    Checks Blob Storage for a valid, unexpired auth token.
    Returns the token string if valid, otherwise None.
    """
    if not token_cache_container_client:
        logger.error("Token cache client is not initialized.")
        return None
        
    try:
        logger.info("Checking for cached auth token...")
        blob_client = token_cache_container_client.get_blob_client(TOKEN_CACHE_BLOB_NAME)
        
        blob_data = blob_client.download_blob().readall()
        token_data = json.loads(blob_data)
        
        token = token_data.get("token")
        expires_str = token_data.get("expires")
        
        if not token or not expires_str:
            logger.warning("Cached token file is corrupt (missing token or expires).")
            return None

        expiry_string_truncated = expires_str[:26] + "Z"
        expires_datetime = datetime.datetime.strptime(expiry_string_truncated, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        safe_expiry_time = expires_datetime - datetime.timedelta(minutes=5)
        
        if datetime.datetime.utcnow() < safe_expiry_time:
            logger.info("Valid, unexpired auth token found in cache.")
            return token
        else:
            logger.info("Cached auth token has expired.")
            return None

    except ResourceNotFoundError:
        logger.info("No auth token cache blob found. Will log in.")
        return None
    except Exception as e:
        logger.error(f"Error reading token cache: {e}", exc_info=True)
        return None

async def _login_and_cache_token() -> Optional[str]:
    """
    Logs in to the job API, gets a new token, and saves it to blob storage.
    Returns the new token string, or None on failure.
    """
    if not JOB_API_BASE_URL or not JOB_API_USER or not JOB_API_PASSWORD:
        logger.error("Job API credentials (URL, User, Pass) are not configured.")
        return None
        
    login_url = f"{JOB_API_BASE_URL}/api/login"
    login_payload = {
        "UserId": JOB_API_USER,
        "Password": JOB_API_PASSWORD
    }
    
    try:
        logger.info(f"Attempting to log in to Job API at {login_url}...")
        response = requests.post(login_url, json=login_payload, timeout=10)
        response.raise_for_status()
        
        response_data = response.json()
        
        token = response_data.get("Token", {}).get("Token")
        expires_str = response_data.get("Token", {}).get("Expires")
        
        if not token or not expires_str:
            logger.error(f"Job API login response is invalid. Response: {response_data}")
            return None
            
        logger.info("Successfully logged in and received new auth token.")
        
        if token_cache_container_client:
            try:
                token_data_to_cache = json.dumps({"token": token, "expires": expires_str})
                blob_client = token_cache_container_client.get_blob_client(TOKEN_CACHE_BLOB_NAME)
                blob_client.upload_blob(token_data_to_cache, overwrite=True)
                logger.info("Successfully cached new auth token to blob storage.")
            except Exception as e:
                logger.error(f"Failed to write new token to cache: {e}", exc_info=True)
        
        return token

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Job API login: {e}", exc_info=True)
        return None

async def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the details for a specific job from the live API,
    handling token authentication and caching automatically.
    """
    token = await _get_cached_token()
    
    if not token:
        logger.info("No valid token. Attempting new login...")
        token = await _login_and_cache_token()
        
    if not token:
        logger.error("Failed to get auth token. Cannot fetch job details.")
        return None
        
    job_url = f"{JOB_API_BASE_URL}/api/Job/GetDetailJob?jobId={job_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        logger.info(f"Fetching job details for {job_id}...")
        response = requests.get(job_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        response_data = response.json()
        
        if response_data.get("status") == 200 and "data" in response_data:
            data = response_data.get("data")
            
            # Handle case where API returns a list instead of a single dict
            if isinstance(data, list):
                if len(data) > 0:
                    logger.info(f"Successfully fetched job details for {job_id} (from list, taking first item).")
                    return data[0]  # Take the first item
                else:
                    logger.error(f"Job API returned an empty list for {job_id}.")
                    return None
            elif isinstance(data, dict):
                logger.info(f"Successfully fetched job details for {job_id}.")
                return data
            else:
                logger.error(f"Job API returned unexpected data type for {job_id}: {type(data)}. Data: {data}")
                return None
        else:
            logger.error(f"Job API returned an error or invalid data for {job_id}. Response: {response_data}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Job API get details: {e}", exc_info=True)
        if e.response is not None and e.response.status_code == 401:
            logger.warning("Got 401 Unauthorized. Token may be invalid. Forcing new login on next attempt.")
            try:
                blob_client = token_cache_container_client.get_blob_client(TOKEN_CACHE_BLOB_NAME)
                blob_client.delete_blob()
                logger.info("Deleted invalid cached token due to 401 error.")
            except Exception as del_e:
                logger.error(f"Failed to delete invalid cached token: {del_e}")
        return None