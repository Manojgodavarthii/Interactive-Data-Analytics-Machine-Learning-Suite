import re, sys
sys.stdout.reconfigure(encoding='utf-8')

files = [
    'modules/correlation_analysis.py',
    'modules/statistical_analysis.py',
    'modules/advanced_analytics.py',
    'modules/data_cleaning.py',
    'modules/ai_insights.py',
    'modules/feature_analysis.py',
    'modules/dataset_overview.py',
    'modules/export.py',
    'modules/report_generation.py',
    'modules/raw_dataset.py',
    'modules/visualizations.py',
    'modules/custom_analysis.py',
]

for f in files:
    src = open(f, encoding='utf-8').read()
    lines = src.split('\n')
    issues = []
    for i, l in enumerate(lines):
        ls = l.strip()
        if 'st.markdown' in ls and '<div' in ls and '</div>' not in ls and 'unsafe_allow_html' in ls:
            issues.append(f'  L{i+1} [DIV]')
        if 'class="card"' in ls:
            issues.append(f'  L{i+1} [CARD]')
    if issues:
        print(f'{f}:')
        for iss in issues:
            print(iss)
        print()
