import pandas as pd
import os

def get_next_version(directory, base_name="submission"):
    os.makedirs(directory, exist_ok=True)
    existing = os.listdir(directory)
    v = 1
    while f"{base_name}_v{v}.csv" in existing:
        v += 1
    return f"{base_name}_v{v}.csv"

def generate():
    source = 'submission.csv'
    target_dir = 'data/submissions'
    
    if not os.path.exists(source):
        print(f"Error: {source} not found. Ensure 'main.py' finished.")
        return
    
    # LOAD AND SORT: This is the fix for the 87% score
    df = pd.read_csv(source)
    df['id'] = pd.to_numeric(df['id'])
    df = df.sort_values('id').reset_index(drop=True)
    
    # Save versioned copy
    filename = get_next_version(target_dir)
    full_path = os.path.join(target_dir, filename)
    df[['id', 'label']].to_csv(full_path, index=False)
    
    # Save the 'Leaderboard Ready' copy to root
    df[['id', 'label']].to_csv('final_submission.csv', index=False)
    
    print(f"✅ Created {full_path}")
    print(f"✅ Created final_submission.csv in root")
    print(df.head(5))

if __name__ == "__main__":
    generate()
