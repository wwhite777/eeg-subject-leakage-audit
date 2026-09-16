#!/usr/bin/env python3
"""Map section headings and table/figure captions to the margin line numbers of main_highlighted.pdf (lineno package).
Output: markdown table (heading -> page, line). Margin numbers are digit-only words left of the main text edge.
Section headings are recognised as ENTIRELY BOLD lines whose text starts with the heading title (the section number is
typeset in a separate block by iopjournal.cls); captions/other phrases are matched by prefix on any line."""
import re, sys
import fitz
pdf = sys.argv[1]; phrases = sys.argv[2:]
doc = fitz.open(pdf)
heads = ["1 Introduction", "2 Methods", "2.1 Datasets", "2.2 Preprocessing", "2.3 Decoders", "2.4 Protocol ladder", "2.5 Controlled analyses",
         "2.6 Statistical analysis", "2.7 Reproducibility", "3 Results", "3.1 The protocol ladder on EEGBCI", "3.2 The ladder on BCI-IV-2a",
         "3.3 Regularization does not remove", "3.4 Where the subject information lives", "3.5 The reference mean", "3.6 Feature dimension and training-subject",
         "3.7 Network width and matched preprocessing", "3.8 Calibration", "3.9 Exploratory", "4 Discussion", "5 Conclusion", "Acknowledgments", "Funding",
         "Data availability", "Disclosure of AI-assisted tools", "References"] + phrases
rows = []
for pno, page in enumerate(doc):
    words = page.get_text("words")
    alpha = [w for w in words if re.search(r"[A-Za-z]{3,}", w[4])]
    if not alpha:
        continue
    xs = sorted(w[0] for w in alpha); edge = xs[len(xs) // 20]  # 5th percentile of text x0
    margin = [(w[1], int(w[4])) for w in words if w[4].isdigit() and w[2] < edge - 2 and 1 <= len(w[4]) <= 4]
    d = page.get_text("dict")
    for block in d["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            if not spans or spans[0]["bbox"][0] < edge - 30:
                continue
            line_text = " ".join(sp["text"].strip() for sp in spans); y = line["bbox"][1]
            # iopjournal.cls: section titles in CMBX (bold flag 16), subsection titles in CMTI (italic flag 2); body lines are regular
            bold = all((sp["flags"] & 16) or (sp["flags"] & 2) for sp in spans) and len(line_text) < 90
            norm = re.sub(r"^[\d.]+\s+", "", line_text).lower()
            for h in heads:
                hn = re.sub(r"^[\d.]+\s+", "", h).lower()
                numbered = bool(re.match(r"^[\d.]+\s+", h))
                hit = (bold and norm.startswith(hn)) if numbered else (line_text.lower().startswith(h.lower()) or (bold and norm.startswith(hn)))
                if hit and h not in [r[0] for r in rows] and len(line_text) < 90:
                    near = [m for m in margin if abs(m[0] - y) < 5]
                    rows.append((h, pno + 1, near[0][1] if near else "n/a"))
order = {h: i for i, h in enumerate(heads)}
rows.sort(key=lambda r: (r[1], r[2] if isinstance(r[2], int) else 10**6, order[r[0]]))
print("| Heading / caption | page | line |"); print("|---|---|---|")
for h, p, n in rows: print(f"| {h} | {p} | {n} |")
print(f"\n(margin numbers detected on page 6: {len([1 for w in doc[5].get_text('words') if w[4].isdigit()])} digit words)")
