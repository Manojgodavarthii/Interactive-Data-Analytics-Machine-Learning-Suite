"""Replace class="card" with proper inline styling in all modules."""
import re

FILES = [
    'modules/data_cleaning.py',
    'modules/ai_insights.py', 
    'modules/feature_analysis.py',
]

# Replace class="card" with inline white card style
CARD_INLINE = (
    'style="background:white;border-radius:12px;padding:1rem 1.2rem;'
    'margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);'
    'border:1px solid #e4e8f0;"'
)

for path in FILES:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Replace class="card" style="..." - keep the existing inline style
    content = re.sub(
        r'class="card"\s+style="([^"]*)"',
        lambda m: f'style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;{m.group(1)}"',
        content
    )
    
    # Replace standalone class="card" with no extra style
    content = content.replace(
        'class="card"',
        'style="background:white;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(0,0,0,0.05);border:1px solid #e4e8f0;"'
    )
    
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        count = original.count('class="card"')
        print(f'  FIXED {path}: {count} class="card" replaced')
    else:
        print(f'  OK    {path}')

print('Done.')
