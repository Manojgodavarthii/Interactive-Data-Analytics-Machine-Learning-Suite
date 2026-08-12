import os

files = [
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

for f in files:
    try:
        src = open(f, encoding='utf-8').read()
        issues = []
        lines = src.split('\n')

        # Check for broken div opening/closing pattern
        broken_open = sum(1 for l in lines if 'st.markdown' in l and '<div' in l and '</div>' not in l and l.strip().endswith("', unsafe_allow_html=True)") == False)
        div_only_open = [l.strip() for l in lines if "st.markdown('<div" in l or 'st.markdown("<div' in l]
        div_only_close = [l.strip() for l in lines if "st.markdown('</div" in l or 'st.markdown("</div' in l]

        if div_only_open:
            issues.append(f'BROKEN DIV OPEN: {len(div_only_open)} standalone <div> markdowns')
        if div_only_close:
            issues.append(f'BROKEN DIV CLOSE: {len(div_only_close)} standalone </div> markdowns')

        # Check for invisible text
        if '-webkit-text-fill-color:transparent' in src or '-webkit-text-fill-color: transparent' in src:
            issues.append('INVISIBLE TEXT: -webkit-text-fill-color:transparent found')

        # Check plotly chart config
        chart_count = src.count('st.plotly_chart(')
        config_count = src.count('displayModeBar')
        missing = chart_count - config_count
        if missing > 0:
            issues.append(f'MODEBAR: {missing}/{chart_count} charts missing displayModeBar config')

        # Check for card/insight-box class usage
        if '.card' in src or 'class="card"' in src:
            card_class = src.count('class="card"')
            if card_class > 0:
                issues.append(f'CARD CLASS: {card_class} usages of class="card" (may be invisible due to transparent CSS)')

        if issues:
            print(f'ISSUES  {f}:')
            for iss in issues:
                print(f'        - {iss}')
        else:
            print(f'OK      {f}')

    except Exception as e:
        print(f'ERROR   {f}: {e}')

print('\nAudit complete.')
