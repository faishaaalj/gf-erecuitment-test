# Bulk Application Submission

This directory contains scripts for bulk processing and submitting applicants to the Azure Function for scoring and indexing.

## Scripts

### 1. `bulk_submit_applicants.py`
Submits applicants in batches to the Azure Function endpoint.

**Features:**
- Async/concurrent processing (10 applicants at a time by default)
- Automatic retries with exponential backoff
- Progress logging
- Saves results and failed applicants for retry

**Usage:**
```bash
python bulk_submit_applicants.py <path_to_applicants.json>
```

**Example:**
```bash
# Local testing
python scripts/bulk_submit_applicants.py data_dummy/applicants.json

# Production (update FUNCTION_ENDPOINT in script first)
python scripts/bulk_submit_applicants.py data_dummy/applicants.json
```

**Configuration (edit script):**
```python
FUNCTION_ENDPOINT = "http://localhost:7071/api/apply_job"  # Change to Azure URL
BATCH_SIZE = 10  # Concurrent submissions
DELAY_BETWEEN_BATCHES = 2  # seconds
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300  # 5 minutes
```

**Output Files:**
- `bulk_submit_YYYYMMDD_HHMMSS.log` - Detailed log
- `submission_results_YYYYMMDD_HHMMSS.json` - All submission results
- `failed_applicants_YYYYMMDD_HHMMSS.json` - Failed submissions for retry

---

### 2. `check_submission_status.py`
Monitors the status of submitted Durable Function orchestrations.

**Usage:**
```bash
python check_submission_status.py <submission_results_file.json> [check_interval] [max_checks]
```

**Example:**
```bash
# Check every 10 seconds, up to 60 times (10 minutes total)
python scripts/check_submission_status.py submission_results_20250810_143022.json 10 60
```

**Output:**
- Real-time status updates in console
- `final_results_YYYYMMDD_HHMMSS.json` - Final status of all submissions

---

## Workflow

### Full Bulk Processing Flow:

1. **Prepare your data** in JSON format (see `data_dummy/applicants.json`)

2. **Start your Azure Function locally** or deploy to Azure:
   ```bash
   func start
   ```

3. **Submit applications in bulk:**
   ```bash
   python scripts/bulk_submit_applicants.py data_dummy/applicants.json
   ```
   This returns immediately with orchestration instance IDs.

4. **Monitor progress:**
   ```bash
   python scripts/check_submission_status.py submission_results_20250810_143022.json
   ```

5. **Retry failed submissions** (if any):
   ```bash
   python scripts/bulk_submit_applicants.py failed_applicants_20250810_143022.json
   ```

---

## Best Practices

### For Production Use:

1. **Update Endpoint:**
   ```python
   # In bulk_submit_applicants.py
   FUNCTION_ENDPOINT = "https://your-function-app.azurewebsites.net/api/apply_job?code=YOUR_FUNCTION_KEY"
   ```

2. **Adjust Batch Size** based on:
   - Azure Function plan (Consumption vs Premium)
   - Document Intelligence API rate limits
   - OpenAI API rate limits
   
   Recommended:
   - **Consumption Plan:** `BATCH_SIZE = 5`, `DELAY_BETWEEN_BATCHES = 5`
   - **Premium Plan:** `BATCH_SIZE = 10-20`, `DELAY_BETWEEN_BATCHES = 2`

3. **Handle Large Datasets:**
   - Split very large files (1000+) into smaller chunks
   - Process during off-peak hours
   - Monitor Azure costs (AI Search, OpenAI, Document Intelligence)

4. **Data Validation:**
   - Ensure all required fields exist: `jobId`, `candidateId`, `cv` (URL)
   - Validate CV URLs are accessible
   - Clean data beforehand to avoid 400 errors

5. **Error Handling:**
   - Review `failed_applicants_*.json` for patterns
   - Common issues: invalid CV URLs, missing fields, network timeouts
   - Retry with exponential backoff for transient errors

---

## Performance Estimates

Based on average processing time per applicant:
- **CV Analysis:** ~5-10 seconds
- **AI Scoring:** ~10-15 seconds  
- **Embedding Generation:** ~2-5 seconds
- **Indexing:** ~1-2 seconds

**Total per applicant:** ~20-35 seconds

**For 1000 applicants:**
- With `BATCH_SIZE=10`: ~35-60 minutes
- With `BATCH_SIZE=5`: ~60-120 minutes

---

## Alternative: Direct Database/Index Update

For **very large datasets** (10,000+) where speed is critical and you already have pre-calculated scores:

1. **Skip orchestration** and directly call `IndexApplicationActivity`
2. **Use Azure Search Indexer** for bulk import
3. **Pre-process CVs offline** and store content

Contact your Azure administrator for assistance with these advanced scenarios.

---

## Troubleshooting

**Issue:** Timeout errors  
**Solution:** Increase `REQUEST_TIMEOUT`, reduce `BATCH_SIZE`

**Issue:** Rate limiting (429 errors)  
**Solution:** Increase `DELAY_BETWEEN_BATCHES`, reduce `BATCH_SIZE`

**Issue:** Many failed submissions  
**Solution:** Check logs, validate data format, test with small batch first

**Issue:** Orchestrations stuck in "Running"  
**Solution:** Check Azure Function logs, verify Durable Functions storage connection
