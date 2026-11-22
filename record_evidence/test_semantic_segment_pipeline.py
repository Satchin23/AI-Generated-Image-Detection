import sys, os
from PIL import Image

sys.path.append(r'd:\Mini_Project')
from views.celebrity_analysis import analyze_semantic_regions_pipeline, MALE_SAMPLE_PATH, FEMALE_SAMPLE_PATH

test_cases = [
    ('Male Celebrity (Virat Kohli)', MALE_SAMPLE_PATH),
    ('Female Celebrity (Sai Pallavi)', FEMALE_SAMPLE_PATH),
    ('Synthetic AI Portrait (Midjourney Sample)', r'd:\Mini_Project\data\raw\AI-GENERATED\midjourney\midjourney_0000.png')
]

for title, path in test_cases:
    print('='*75)
    print('TEST CASE:', title)
    img = Image.open(path).convert('RGB')
    res = analyze_semantic_regions_pipeline(img, os.path.basename(path))
    
    w = res['whole']
    m = res['metrics']
    print(f"Whole Image Baseline: {w['label']} (Conf={w['confidence']*100:.1f}%, Real={w['real_prob']*100:.1f}%, AI={w['ai_prob']*100:.1f}%)")
    print(f"Estimated Overall Originality: {m['estimated_originality_pct']}%")
    print(f"Estimated Overall AI-Generated: {m['estimated_ai_pct']}%")
    print(f"Mean Segment Originality: {m['mean_originality_pct']}%")
    sr = m['strongest_real_region']
    sa = m['strongest_ai_region']
    print(f"Strongest REAL Region: {sr['name']} ({sr['real_prob']*100:.1f}%)")
    print(f"Strongest AI-INDICATIVE Region: {sa['name']} ({sa['ai_prob']*100:.1f}%)")
    print('Individual Segment Breakdown:')
    for r in res['regions']:
        print(f"  - {r['name']:<14}: {r['label']:<12} | Conf: {r['confidence']*100:5.1f}% | Real: {r['real_prob']*100:5.1f}% | AI: {r['ai_prob']*100:5.1f}% | Area: {r['area_pct']:4.1f}% | Lean: {r['lean']} (+{r['strength']:.1f}%)")
