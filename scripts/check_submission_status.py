"""
Check the status of bulk submissions.
Monitors Durable Function orchestration instances.
"""

import json
import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def check_status(session: aiohttp.ClientSession, status_uri: str) -> Dict[str, Any]:
    """Check the status of a single orchestration."""
    try:
        async with session.get(status_uri, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"runtimeStatus": "Unknown", "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"runtimeStatus": "Error", "error": str(e)}


async def monitor_submissions(results_file: str, check_interval: int = 10, max_checks: int = 60):
    """
    Monitor the status of submitted applications.
    
    Args:
        results_file: Path to the submission_results_*.json file
        check_interval: Seconds between status checks
        max_checks: Maximum number of checks before giving up
    """
    # Load submission results
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Filter for submitted applications
    submitted = [r for r in results if r['status'] == 'submitted' and r.get('statusQueryGetUri')]
    
    if not submitted:
        logger.warning("No submitted applications found to monitor.")
        return
    
    logger.info(f"Monitoring {len(submitted)} submitted applications...")
    
    pending = submitted.copy()
    completed_results = []
    
    for check_num in range(1, max_checks + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Status Check #{check_num} - {len(pending)} applications pending")
        logger.info(f"{'='*80}")
        
        async with aiohttp.ClientSession() as session:
            tasks = [check_status(session, r['statusQueryGetUri']) for r in pending]
            statuses = await asyncio.gather(*tasks)
        
        still_pending = []
        
        for i, status_data in enumerate(statuses):
            result = pending[i]
            runtime_status = status_data.get('runtimeStatus', 'Unknown')
            
            if runtime_status == 'Completed':
                output = status_data.get('output', {})
                result['finalStatus'] = 'Completed'
                result['applicationId'] = output.get('applicationId')
                result['score'] = output.get('score')
                result['indexed'] = output.get('indexed', True)
                result['operation'] = output.get('operation')
                completed_results.append(result)
                logger.info(f"✓ Completed: Candidate {result['candidateId']} - Score: {result.get('score', 'N/A')}")
                
            elif runtime_status == 'Failed':
                result['finalStatus'] = 'Failed'
                result['error'] = status_data.get('output', {}).get('error', 'Unknown error')
                completed_results.append(result)
                logger.error(f"✗ Failed: Candidate {result['candidateId']} - {result['error']}")
                
            elif runtime_status in ['Running', 'Pending']:
                still_pending.append(result)
                logger.info(f"⏳ Pending: Candidate {result['candidateId']} - Status: {runtime_status}")
                
            else:
                result['finalStatus'] = runtime_status
                completed_results.append(result)
                logger.warning(f"? Unknown: Candidate {result['candidateId']} - Status: {runtime_status}")
        
        pending = still_pending
        
        if not pending:
            logger.info("\n✓ All applications completed!")
            break
        
        if check_num < max_checks:
            logger.info(f"\nWaiting {check_interval}s before next check...")
            await asyncio.sleep(check_interval)
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    
    completed_count = sum(1 for r in completed_results if r.get('finalStatus') == 'Completed')
    failed_count = sum(1 for r in completed_results if r.get('finalStatus') == 'Failed')
    indexed_count = sum(1 for r in completed_results if r.get('indexed') == True)
    skipped_count = sum(1 for r in completed_results if r.get('indexed') == False)
    
    logger.info(f"Total Monitored: {len(submitted)}")
    logger.info(f"Completed Successfully: {completed_count}")
    logger.info(f"  - Indexed: {indexed_count}")
    logger.info(f"  - Skipped (below threshold): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Still Pending: {len(pending)}")
    
    # Save final results
    final_results_file = f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_results_file, 'w', encoding='utf-8') as f:
        json.dump(completed_results + pending, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFinal results saved to: {final_results_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_submission_status.py <submission_results_file.json> [check_interval] [max_checks]")
        print("Example: python check_submission_status.py submission_results_20250810_143022.json 10 60")
        sys.exit(1)
    
    results_file = sys.argv[1]
    check_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_checks = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    
    asyncio.run(monitor_submissions(results_file, check_interval, max_checks))


if __name__ == "__main__":
    main()
