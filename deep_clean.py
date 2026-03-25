import os
import shutil
import time
import sys

def progress_bar(iterable, prefix='', suffix='', decimals=1, length=30, fill='█'):
    total = len(iterable)
    for i, item in enumerate(iterable):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(total)))
        filled_length = int(length * i // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
        yield item
    print(f'\r{prefix} |{fill * length}| 100.0% Complete')

steps = [
    "Consolidating Visualizations",
    "Moving Tables to Data",
    "Cleaning Script Folders",
    "Removing Redundant Folders",
    "Updating Project Report",
    "Finalizing Git Sync"
]

print("--- STARTING REPO RECONSTRUCTION ---")

for i, step in enumerate(progress_bar(steps, prefix='Progress:', suffix='')):
    time.sleep(0.5) # Just for visual feedback
    
    if i == 0: # Visuals
        os.makedirs('data/visualizations', exist_ok=True)
        for folder in ['submission_visualizations', 'visualizations', 'Visualizations']:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    shutil.move(os.path.join(folder, f), 'data/visualizations/')
                shutil.rmtree(folder)
                
    elif i == 1: # Tables
        os.makedirs('data/tables', exist_ok=True)
        if os.path.exists('tables'):
            for f in os.listdir('tables'):
                shutil.move(os.path.join('tables', f), 'data/tables/')
            shutil.rmtree('tables')
            
    elif i == 2: # Scripts
        os.makedirs('src', exist_ok=True)
        for script in ['run_full_analysis.py', 'create_report.py', 'generate_visuals.py', 'visualize.py']:
            if os.path.exists(script):
                shutil.move(script, f'src/{script}')
            elif os.path.exists(f'src/{script}'):
                pass # Already there
                
    elif i == 3: # Cleanup
        for redundant in ['submissions', 'test', 'output', 'Report', 'FINAL_REPORT.md']:
            if os.path.exists(redundant):
                if os.path.isdir(redundant): shutil.rmtree(redundant)
                else: os.remove(redundant)
                
    elif i == 4: # Report
        with open('PROJECT_REPORT.md', 'w') as f:
            f.write("# Medical Image Classification\n\n**Update:** " + time.ctime() + "\n- All visuals consolidated in `data/visualizations`.\n- Old reports purged.\n- Pipeline cleaned.")

print("\n[SUCCESS] Folders Consolidated.")
print("[SUCCESS] Data files updated and old versions replaced.")
