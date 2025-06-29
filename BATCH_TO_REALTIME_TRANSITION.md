# 🔄 **Complete Guide: Switching from Batch to Real-Time Processing**

This guide covers the complete process of safely transitioning from batch processing to real-time processing, including background process management.

## 📋 **Prerequisites**

- You have a running batch process: `nohup python batch_manager.py --create`
- You want to switch to real-time processing
- You want to run real-time processing in the background

---

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

## 🔍 **Step 3: Analyze Transition Readiness**

```bash
# Analyze if the system is ready for transition
python batch_manager.py --analyze-transition
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

## 🔄 **Step 4: Execute the Transition**

You have two options depending on your analysis results:

### Option A: Wait for Active Jobs (Recommended)
If you have few active jobs and want to preserve their progress:

```bash
# Wait for active jobs to complete, then transition
python batch_manager.py --switch-to-realtime
```

### Option B: Force Immediate Transition
If you want to switch immediately and cancel active jobs:

```bash
# Force transition (cancels active batch jobs)
python batch_manager.py --switch-to-realtime --force
```

**Expected Output:**
```
🚀 Switching from batch to real-time processing...
============================================================

🔄 Executing transition (force=false)...

📋 TRANSITION RESULTS
==============================
Status: completed
✅ Transition completed successfully!
⏱️  Elapsed Time: 45.23 seconds

📊 FINAL STATE:
   • Total processed hashes: 1,234,567
   • From batch only: 1,200,000
   • From real-time only: 34,567

🚀 Real-time processing is now active!
💡 You can now run: python main.py --config config.yml
```

---

## ⚡ **Step 5: Run Real-Time Processing in Background**

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

## 📝 **Step 6: Monitor Real-Time Processing**

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
# Check what went wrong
python batch_manager.py --analyze-transition

# Force transition if safe to do so
python batch_manager.py --switch-to-realtime --force

# Or check batch status
python batch_manager.py --status
```

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

## ✅ **Quick Reference Commands**

```bash
# Complete transition workflow
kill [BATCH_PID]                                    # Stop batch process
python batch_manager.py --status                    # Check status  
python batch_manager.py --analyze-transition        # Analyze readiness
python batch_manager.py --switch-to-realtime        # Execute transition
nohup python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing > embedding_realtime.log 2>&1 &

# Process management
ps aux | grep "main.py"                            # Find process
tail -f embedding_realtime.log                     # Monitor logs
kill [REALTIME_PID]                                # Stop process
python main.py --status                            # Check pipeline status
```

This complete workflow ensures a safe, monitored transition from batch to real-time processing with full background operation support!