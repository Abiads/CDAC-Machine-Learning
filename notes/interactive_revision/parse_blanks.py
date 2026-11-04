import json, re

with open(r'c:\Users\conne\Downloads\ML_CDAC_ALILOG\notes\Fill_in_the_Blanks_and_Conceptual_Review_Guide.txt', 'r', encoding='utf-8') as f:
    text = f.read()

aliases_map = {
    "1.1": ["two-dimensional", "2d", "two dimensional", "2-dimensional"],
    "1.2": ["rows, columns", "rows and columns", "row, column", "rows,columns", "rows columns"],
    "1.3": ["key-value", "key value", "keys and values", "key:value", "key-value pairs"],
    "1.4": ["immutable", "unmutable", "not mutable"],
    "1.5": ["data analysis", "exploratory data analysis", "eda", "data exploration"],
    "1.6": ["columns", "column", "attributes", "features"],
    "1.7": ["database", "db", "relational database", "sql database"],
    "1.8": ["dropna", "dropna()", "df.dropna", "df.dropna()"],
    "2.1": ["categorical", "qualitative", "categorical data", "qualitative data"],
    "2.2": ["continuous", "continuous data", "continuous feature"],
    "2.3": ["discrete", "discrete data", "discrete feature", "count"],
    "3.1": ["skewness", "skew", "asymmetry"],
    "3.2": ["68-95-99.7 rule", "empirical rule", "68-95-99.7", "three sigma rule", "3-sigma rule", "empirical"],
    "3.3": ["leptokurtic", "positive kurtosis", "leptokurtic distribution"]
}

section_sources = {
    "1": ("Statistics Session 1: DataFrames in Python (Pandas)", "Session 1 Dataframes in Python 2 aug 2025 -Tuned.pptx (Slides 9-10)"),
    "2": ("Statistics Session 4: Understanding Data Features", "Session 4 Understanding Data Features -Tuned.pptx (Slide 21)"),
    "3": ("Statistics Session 5: Data Distributions", "Session 5 Data Distributions - Tuned.pptx (Slides 33-34)")
}

q_blocks = re.findall(r'QUESTION (\d+\.\d+):\s*\n(.*?)\n\s*ANSWER:\s*([^\n]+)\s*\n\s*IN-DEPTH EXPLANATION:\s*\n(.*?)(?=(?:QUESTION \d+\.\d+:|---+|SECTION \d+:|\Z))', text, re.DOTALL)

items = []
for q_id, raw_prompt, ans, expl in q_blocks:
    sec_num = q_id.split('.')[0]
    sec_title, sec_source = section_sources.get(sec_num, (f"Section {sec_num}", ""))
    clean_prompt = ' '.join(raw_prompt.strip().replace('"', '').split())
    items.append({
        "id": q_id,
        "sectionNum": int(sec_num),
        "sectionTitle": sec_title,
        "source": sec_source,
        "prompt": clean_prompt,
        "answer": ans.strip(),
        "aliases": aliases_map.get(q_id, [ans.strip().lower()]),
        "explanation": expl.strip()
    })

print(f'Parsed all {len(items)} blanks items perfectly!')
with open(r'c:\Users\conne\Downloads\ML_CDAC_ALILOG\notes\interactive_revision\blanks.json', 'w', encoding='utf-8') as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print('Saved to blanks.json')
