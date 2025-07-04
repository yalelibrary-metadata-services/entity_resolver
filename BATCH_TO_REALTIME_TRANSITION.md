# 🔄 **Complete Guide: Bidirectional Processing Mode Transitions**

This guide covers the complete process of safely transitioning between batch and real-time processing modes, including background process management.

## 📋 **Prerequisites**

### For Batch → Real-Time Transition
- You have a running batch process: `nohup python batch_manager.py --create`
- You want to switch to real-time processing
- You want to run real-time processing in the background

### For Real-Time → Batch Transition  
- You have a running real-time process: `nohup python main.py --start embedding_and_indexing`
- You want to switch to batch processing for cost efficiency
- You want better control over large-scale processing

---

## 🎯 **Transition Workflows**

Choose your transition path:

### 🔄 **Option A: Batch → Real-Time** (Original workflow)
**Use when:** You want faster processing and immediate results
**Benefits:** Real-time feedback, faster individual requests
**Considerations:** Higher per-token cost, daily rate limits

### 🔄 **Option B: Real-Time → Batch** (New workflow)  
**Use when:** You want cost efficiency and can wait for results
**Benefits:** 50% cost reduction, larger batch processing capacity
**Considerations:** 24-hour processing delays, less real-time feedback

---

## 🔄 **BATCH → REAL-TIME TRANSITION**

## 🛑 **Step 1: Stop the Running Batch Process**

### Find the Batch Process
```bash
# Find the running batch process
ps aux | grep "batch_manager.py --create"
```

Example output:
```
user    12345  0.1  0.5  123456  67890 ?  S  10:30  0:05 python batch_manager.py --create
```

### Kill the Process
```bash
# Replace 12345 with the actual PID from above
kill 12345

# If it doesn't stop gracefully within 30 seconds, force kill:
kill -9 12345
```

### Verify Process is Stopped
```bash
# Check that the process is gone
ps aux | grep "batch_manager.py --create"
# Should show only the grep command itself
```

---

## 📊 **Step 2: Check Current Batch Status**

```bash
# Check what batch jobs are currently active
python batch_manager.py --status
```

**Expected Output:**
```
📊 Checking batch job status...
🔍 GRANULAR STATUS BREAKDOWN:
   ⏳ Pending: 5
   🔄 In Progress: 3
   ✅ Completed: 12
   ❌ Failed: 2
```

**Note the numbers** - you'll want to know how many jobs are active before transitioning.

---

## 📥 **Step 3: Download Completed Results**

```bash
# Download and process any completed batch jobs
python batch_manager.py --download
```

**Expected Output:**
```
📥 Processing completed batch jobs...
🎉 Batch processing completed successfully!
   Weaviate collection now contains 1,234,567 objects
```

**Note:** This step ensures that all completed batch work is properly downloaded and processed before transitioning. If no jobs are completed yet, you'll see a message indicating no completed jobs are available.

---

## 🔍 **Step 4: Analyze Transition Readiness**

```bash
# Analyze if the system is ready for transition
python src/transition_controller.py --analyze-only --direction batch_to_realtime
```

**Expected Output:**
```
🔍 Analyzing batch-to-real-time transition readiness...
============================================================
Transition Feasible: ✅ YES

📊 BATCH PROCESSING STATE:
   • Processed hashes: 1,234,567
   • Failed requests: 45
   • Active jobs: 8

💡 RECOMMENDATIONS:
   • Wait for 8 active batch jobs to complete before transition
   • 45 failed requests will be migrated for real-time retry
```

---

## 🔄 **Step 5: Execute the Transition**

You have two options depending on your analysis results:

### Option A: Wait for Active Jobs (Recommended)
If you have few active jobs and want to preserve their progress:

```bash
# Wait for active jobs to complete, then transition
python src/transition_controller.py --direction batch_to_realtime
```

### Option B: Force Immediate Transition
If you want to switch immediately and cancel active jobs:

```bash
# Force transition (cancels active batch jobs)
python src/transition_controller.py --direction batch_to_realtime --force
```

**Expected Output:**
```
============================================================
BATCH-TO-REALTIME TRANSITION RESULTS
============================================================
Status: completed
Elapsed Time: 45.23 seconds

Total Processed Hashes: 1,234,567
From Batch Only: 1,200,000
From Real-time Only: 34,567

✅ Transition completed successfully!
Real-time processing is now active with all batch progress preserved.
```

---

## ⚡ **Step 6: Run Real-Time Processing in Background**

### Option A: Run Only Embedding Stage (Recommended)
```bash
# Run only embedding and indexing stage in background
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &
```

### Option B: Run Full Pipeline
```bash
# Run complete pipeline in background
nohup python main.py --config config.yml > pipeline_realtime.log 2>&1 &
```

### Get the Process ID
```bash
# The command above will output something like:
[1] 67890

# Or find it manually:
ps aux | grep "main.py"
```

**Save this PID** - you'll need it to manage the process later.

---

## 📝 **Step 7: Monitor Real-Time Processing**

### Check Process Status
```bash
# Verify the process is running
ps aux | grep "main.py"
```

### Monitor Logs
```bash
# Watch the logs in real-time
tail -f embedding_realtime.log

# Or check recent entries
tail -100 embedding_realtime.log
```

### Check Pipeline Status
```bash
# Check overall pipeline status
python main.py --status
```

---

## 🔧 **Process Management Commands**

### Check Running Process
```bash
# Find your real-time process
ps aux | grep "main.py"
```

### Stop Real-Time Process
```bash
# Graceful stop (replace 67890 with actual PID)
kill 67890

# Force stop if needed
kill -9 67890
```

### Restart Real-Time Process
```bash
# Stop current process
kill [PID]

# Start new process
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &
```

---

## 📊 **Monitoring and Troubleshooting**

### Check Progress
```bash
# View recent log entries
tail -50 embedding_realtime.log

# Search for errors
grep -i error embedding_realtime.log

# Search for progress indicators
grep -i "processed\|completed\|tokens" embedding_realtime.log
```

### Common Log Patterns to Look For
```bash
# Success indicators
grep "Successfully processed" embedding_realtime.log
grep "Vector processed" embedding_realtime.log

# Rate limiting (normal)
grep "rate limit\|TPD limit" embedding_realtime.log

# Errors (investigate these)
grep -i "error\|failed\|exception" embedding_realtime.log
```

### Check System Resources
```bash
# Check CPU and memory usage
top -p [PID]

# Or use htop if available
htop -p [PID]
```

---

## ⚠️ **Important Notes**

### **Rate Limiting Behavior**
- Real-time mode respects **500M tokens per day** limit
- When TPD limit reached, process **automatically pauses for 30 minutes**
- This is normal behavior - the process will resume automatically

### **Progress Preservation**
- ✅ All batch progress is preserved during transition
- ✅ Failed batch requests are automatically retried in real-time
- ✅ No duplicate processing occurs
- ✅ Checkpoints are continuously saved

### **Log File Management**
```bash
# Log files can grow large, rotate them periodically:
mv embedding_realtime.log embedding_realtime_$(date +%Y%m%d_%H%M%S).log
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &
```

---

## 🚨 **Emergency Procedures**

### If Transition Fails
```bash
# Check what went wrong (for either direction)
python src/transition_controller.py --analyze-only --direction batch_to_realtime
python src/transition_controller.py --analyze-only --direction realtime_to_batch

# Force transition if safe to do so
python src/transition_controller.py --direction batch_to_realtime --force
python src/transition_controller.py --direction realtime_to_batch --force

# Override job state tracking (for stale job issues)
python src/transition_controller.py --direction batch_to_realtime --override
python src/transition_controller.py --direction realtime_to_batch --override

# Or check current status
python batch_manager.py --status                    # For batch processing
python main.py --status                            # For real-time processing
```

### If Job State Tracking is Stale
```bash
# When batch jobs show as active but are actually completed
# This bypasses job validation and forces consolidation
python src/transition_controller.py --direction realtime_to_batch --override

# Warning: This may skip incomplete job processing
# Use only when certain that "active" jobs are actually stale
```

**When to use `--override`:**
- Batch jobs show as active but have actually completed
- Job tracking state is inconsistent due to process interruptions
- Connection issues prevented proper job status updates
- Need to force transition despite apparent active jobs

**⚠️ Warning:** Override skips job completion processing and may result in loss of incomplete work. Use only when confident that supposedly active jobs are stale.

### If Real-Time Process Crashes
```bash
# Check the log for errors
tail -100 embedding_realtime.log

# Check pipeline status
python main.py --status

# Resume from checkpoint
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing --resume > embedding_realtime.log 2>&1 &
```

### Cancel Everything and Start Over
```bash
# Cancel all batch jobs
python batch_manager.py --cancel

# Reset everything (WARNING: loses progress)
python batch_manager.py --reset

# Start fresh with real-time
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &
```

---

## 🔄 **REAL-TIME → BATCH TRANSITION**

## 🛑 **Step 1: Stop the Running Real-Time Process**

### Find the Real-Time Process
```bash
# Find the running real-time process
ps aux | grep "main.py"
```

Example output:
```
user    67890  0.1  0.5  123456  67890 ?  S  10:30  0:05 python main.py --start embedding_and_indexing
```

### Kill the Process
```bash
# Replace 67890 with the actual PID from above
kill 67890

# If it doesn't stop gracefully within 30 seconds, force kill:
kill -9 67890
```

### Verify Process is Stopped
```bash
# Check that the process is gone
ps aux | grep "main.py"
# Should show only the grep command itself
```

---

## 🔍 **Step 2: Analyze Transition Readiness**

```bash
# Analyze if the system is ready for transition
python src/transition_controller.py --analyze-only --direction realtime_to_batch
```

**Expected Output:**
```
============================================================
PRE-TRANSITION ANALYSIS
============================================================
Transition Feasible: ✅ YES

💡 RECOMMENDATIONS:
   • Real-time processing will be safely terminated
   • All progress will be preserved in batch mode
   • Failed requests will be queued for batch retry
```

---

## 🔄 **Step 3: Execute the Transition**

```bash
# Execute real-time to batch transition
python src/transition_controller.py --direction realtime_to_batch
```

**Expected Output:**
```
============================================================
REALTIME-TO-BATCH TRANSITION RESULTS
============================================================
Status: completed
Elapsed Time: 15.67 seconds

Total Processed Hashes: 1,234,567
From Batch Only: 1,200,000
From Real-time Only: 34,567

✅ Transition completed successfully!
Batch processing is now active with all real-time progress preserved.
```

---

## 📦 **Step 4: Run Batch Processing**

### Start Batch Processing
```bash
# Run batch processing in background
nohup python batch_manager.py --create > batch_processing.log 2>&1 &
```

### Monitor Batch Status
```bash
# Check batch job status
python batch_manager.py --status

# Monitor logs
tail -f batch_processing.log

# Download completed results (when ready)
python batch_manager.py --download
```

---

## ✅ **Quick Reference Commands**

### Batch → Real-Time Transition
```bash
# Complete batch-to-realtime workflow
kill [BATCH_PID]                                    # Stop batch process
python batch_manager.py --status                    # Check batch status  
python batch_manager.py --download                  # Download completed results
python src/transition_controller.py --analyze-only --direction batch_to_realtime  # Analyze readiness
python src/transition_controller.py --direction batch_to_realtime                 # Execute transition
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &
```

### Real-Time → Batch Transition  
```bash
# Complete realtime-to-batch workflow
kill [REALTIME_PID]                                 # Stop real-time process
python src/transition_controller.py --analyze-only --direction realtime_to_batch  # Analyze readiness
python src/transition_controller.py --direction realtime_to_batch                 # Execute transition
nohup python batch_manager.py --create > batch_processing.log 2>&1 &              # Start batch processing
```

### General Process Management
```bash
# Find processes
ps aux | grep "main.py"                            # Find real-time process
ps aux | grep "batch_manager.py"                   # Find batch process

# Monitor logs
tail -f embedding_realtime.log                     # Monitor real-time logs
tail -f batch_processing.log                       # Monitor batch logs

# Check status
python main.py --status                            # Check pipeline status
python batch_manager.py --status                   # Check batch status
```

This complete workflow ensures safe, monitored transitions between batch and real-time processing with full background operation support!