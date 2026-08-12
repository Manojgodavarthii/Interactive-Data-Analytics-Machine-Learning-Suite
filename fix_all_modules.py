"""
Fix all modules:
1. Remove standalone <div class="card"> / </div> broken patterns
2. Add displayModeBar:False to charts missing it
3. Replace invisible -webkit-text-fill-color:transparent
"""
import re

FILES_TO_FIX = [
    'modules/correlation_analysis.py',
    'modules/statistical_analysis.py',
    'modules/advanced_analytics.py',
    'modules/data_cleaning.py',
    'modules/ai_insights.py',
    'modules/feature_analysis.py',
    'modules/custom_analysis.py',
    'modules/dataset_overview.py',
    'modules/export.py',
    'modules/report_generation.py',
    'modules/raw_dataset.py',
    'modules/visualizations.py',
]

def fix_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    removed = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Remove standalone div-open: st.markdown('<div class="card">', unsafe_allow_html=True)
        is_broken_open = (
            re.match(r'''.*st\.markdown\(['"]<div[^>]*>['"],\s*unsafe_allow_html=True\)''', stripped) or
            re.match(r'''.*st\.markdown\(f?['"]<div[^>]*>['"],\s*unsafe_allow_html=True\)''', stripped)
        )

        # Remove standalone div-close: st.markdown('</div>', unsafe_allow_html=True)
        is_broken_close = (
            stripped == "st.markdown('</div>', unsafe_allow_html=True)" or
            stripped == 'st.markdown("</div>", unsafe_allow_html=True)' or
            stripped == "st.markdown('</div>',unsafe_allow_html=True)" or
            stripped == 'st.markdown("</div>",unsafe_allow_html=True)'
        )

        if is_broken_open or is_broken_close:
            removed += 1
            # Add a blank line as replacement to keep structure
            # (only if not already preceded by blank line)
            continue

        # Fix -webkit-text-fill-color:transparent (makes text invisible)
        if '-webkit-text-fill-color:transparent' in line or '-webkit-text-fill-color: transparent' in line:
            line = line.replace('-webkit-text-fill-color:transparent', 'color:#6366f1')
            line = line.replace('-webkit-text-fill-color: transparent', 'color:#6366f1')

        new_lines.append(line)

    # Fix displayModeBar missing in plotly_chart calls
    content = ''.join(new_lines)
    chart_fixes = 0

    def add_config(m):
        nonlocal chart_fixes
        inner = m.group(1)
        if 'displayModeBar' in inner:
            return m.group(0)
        chart_fixes += 1
        return f'st.plotly_chart({inner}, config={{"displayModeBar": False}})'

    content = re.sub(r'st\.plotly_chart\(([^)]+)\)', add_config, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return removed, chart_fixes


print("Fixing all modules...\n")
total_removed = 0
total_charts = 0

for path in FILES_TO_FIX:
    try:
        removed, charts = fix_file(path)
        total_removed += removed
        total_charts += charts
        status = []
        if removed > 0:
            status.append(f'{removed} broken divs removed')
        if charts > 0:
            status.append(f'{charts} charts fixed')
        if status:
            print(f'  FIXED  {path}: {", ".join(status)}')
        else:
            print(f'  OK     {path}')
    except Exception as e:
        print(f'  ERROR  {path}: {e}')

print(f'\nDone. Total: {total_removed} broken divs removed, {total_charts} chart configs added.')
