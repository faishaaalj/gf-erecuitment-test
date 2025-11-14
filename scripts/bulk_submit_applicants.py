"""
Bulk Application Submission Script
Processes applicants from JSON file and submits them to Azure Function for scoring and indexing.
"""

import json
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
from pathlib import Path
import time
from datetime import datetime

# Configuration
FUNCTION_ENDPOINT = "http://localhost:7071/api/apply_job"  # Change to Azure URL in production
BATCH_SIZE = 10  # Process 10 applicants concurrently
DELAY_BETWEEN_BATCHES = 2  # seconds
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300  # 5 minutes per request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'bulk_submit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def submit_application(
    session: aiohttp.ClientSession, 
    applicant: Dict[str, Any], 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Submit a single application to the Azure Function."""
    async with semaphore:
        candidate_id = applicant.get('candidateId', 'unknown')
        job_id = applicant.get('jobId', 'unknown')
        
        # Validate required fields
        if not applicant.get('cv'):
            logger.warning(f"Skipping candidate {candidate_id} for job {job_id}: Missing CV URL")
            return {
                "candidateId": candidate_id,
                "jobId": job_id,
                "status": "skipped",
                "reason": "Missing CV URL"
            }
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Submitting candidate {candidate_id} for job {job_id} (Attempt {attempt + 1}/{MAX_RETRIES})")
                
                async with session.post(
                    FUNCTION_ENDPOINT,
                    json=applicant,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as response:
                    if response.status == 202:  # Accepted (Durable Function started)
                        result = await response.json()
                        logger.info(f"✓ Successfully submitted candidate {candidate_id} for job {job_id}")
                        return {
                            "candidateId": candidate_id,
                            "jobId": job_id,
                            "status": "submitted",
                            "instanceId": result.get('id'),
                            "statusQueryGetUri": result.get('statusQueryGetUri')
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"✗ Failed to submit candidate {candidate_id}: HTTP {response.status} - {error_text}")
                        
                        if response.status == 400:  # Bad request - don't retry
                            return {
                                "candidateId": candidate_id,
                                "jobId": job_id,
                                "status": "failed",
                                "reason": f"Bad request: {error_text}"
                            }
                        
                        # Retry for other errors
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        return {
                            "candidateId": candidate_id,
                            "jobId": job_id,
                            "status": "failed",
                            "reason": f"HTTP {response.status} after {MAX_RETRIES} attempts"
                        }
                        
            except asyncio.TimeoutError:
                logger.error(f"✗ Timeout submitting candidate {candidate_id} (Attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "candidateId": candidate_id,
                    "jobId": job_id,
                    "status": "failed",
                    "reason": "Timeout after retries"
                }
            except Exception as e:
                logger.error(f"✗ Error submitting candidate {candidate_id}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "candidateId": candidate_id,
                    "jobId": job_id,
                    "status": "failed",
                    "reason": str(e)
                }


async def process_batch(
    applicants: List[Dict[str, Any]], 
    batch_num: int, 
    total_batches: int
) -> List[Dict[str, Any]]:
    """Process a batch of applicants concurrently."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Batch {batch_num}/{total_batches} ({len(applicants)} applicants)")
    logger.info(f"{'='*80}")
    
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    connector = aiohttp.TCPConnector(limit=BATCH_SIZE)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [submit_application(session, applicant, semaphore) for applicant in applicants]
        results = await asyncio.gather(*tasks)
    
    return results


async def bulk_submit(applicants_file: str):
    """Main function to process all applicants from JSON file."""
    start_time = time.time()
    
    # Load applicants
    logger.info(f"Loading applicants from {applicants_file}")
    with open(applicants_file, 'r', encoding='utf-8') as f:
        applicants = json.load(f)
    
    total_applicants = len(applicants)
    logger.info(f"Loaded {total_applicants} applicants")
    
    # Split into batches
    batches = [applicants[i:i + BATCH_SIZE] for i in range(0, total_applicants, BATCH_SIZE)]
    total_batches = len(batches)
    
    all_results = []
    
    # Process each batch
    for i, batch in enumerate(batches, 1):
        batch_results = await process_batch(batch, i, total_batches)
        all_results.extend(batch_results)
        
        # Delay between batches to avoid overwhelming the system
        if i < total_batches:
            logger.info(f"Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)
    
    # Summary
    elapsed_time = time.time() - start_time
    submitted_count = sum(1 for r in all_results if r['status'] == 'submitted')
    failed_count = sum(1 for r in all_results if r['status'] == 'failed')
    skipped_count = sum(1 for r in all_results if r['status'] == 'skipped')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Applicants: {total_applicants}")
    logger.info(f"Successfully Submitted: {submitted_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Total Time: {elapsed_time:.2f} seconds")
    logger.info(f"Average Time per Applicant: {elapsed_time/total_applicants:.2f} seconds")
    
    # Save results to file
    results_file = f"submission_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Save failed/skipped for retry
    failed_applicants = [
        applicants[i] for i, r in enumerate(all_results) 
        if r['status'] in ['failed', 'skipped']
    ]
    if failed_applicants:
        failed_file = f"failed_applicants_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_applicants, f, indent=2, ensure_ascii=False)
        logger.info(f"Failed/skipped applicants saved to: {failed_file} (for retry)")


def main():
    """Entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bulk_submit_applicants.py <path_to_applicants_2.json>")
        print("Example: python bulk_submit_applicants.py ../data_dummy/applicants_2.json")
        sys.exit(1)
    
    applicants_file = sys.argv[1]
    
    if not Path(applicants_file).exists():
        print(f"Error: File not found: {applicants_file}")
        sys.exit(1)
    
    # Run async function
    asyncio.run(bulk_submit(applicants_file))


if __name__ == "__main__":
    main()
