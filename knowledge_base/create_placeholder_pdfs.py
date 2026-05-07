"""
Create placeholder PDF files for the knowledge base.
These contain structured pathology descriptions for RCM and ARVC.
"""

from pathlib import Path


def create_placeholder_pdfs(output_dir: str = None):
    """Create placeholder PDF documents with cardiac pathology descriptions."""
    try:
        from fpdf import FPDF
        HAS_FPDF = True
    except ImportError:
        HAS_FPDF = False

    if output_dir is None:
        output_dir = "/home/regouchang/3D-MedDiffusion/knowledge_base"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # RCM content
    rcm_text = """
Restrictive Cardiomyopathy (RCM) - Pathology Reference

Late Gadolinium Enhancement (LGE) Characteristics:
- LGE in RCM typically shows patchy or diffuse enhancement patterns
- Enhancement commonly observed in the subendocardial layer
- May involve the right ventricular insertion points
- Mid-wall fibrosis is a common finding in advanced stages
- Global fibrosis burden correlates with disease severity

Myocardial Tissue Characterization:
- Increased myocardial stiffness with preserved wall thickness
- Bi-atrial dilation secondary to impaired ventricular filling
- Normal or near-normal systolic function in early stages
- Diastolic dysfunction is the hallmark feature

MRI Findings:
- Normal wall thickness with restricted filling pattern
- Atrial enlargement disproportionate to ventricular size
- Pericardial effusion may be present
- T1 mapping shows elevated native T1 values
- ECV (extracellular volume) is typically elevated
"""

    arvc_text = """
Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC) - Pathology Reference

Late Gadolinium Enhancement (LGE) Characteristics:
- LGE in ARVC predominantly affects the right ventricle
- Enhancement typically seen in the RV free wall
- Subtricuspid and infundibular regions commonly involved
- Transmural fibrofatty replacement visible on LGE
- Left ventricular involvement in advanced cases (especially posterolateral wall)
- Epicardial enhancement pattern may be observed

Myocardial Tissue Characterization:
- Fibrofatty infiltration of the RV myocardium
- Progressive wall thinning and aneurysm formation
- Regional wall motion abnormalities
- Fatty metaplasia visible on T1-weighted imaging

MRI Findings:
- RV dilation and regional akinesia
- RV outflow tract enlargement
- Fat infiltration in RV free wall
- Wall motion abnormalities in subtricuspid region
- LV involvement pattern: posterolateral > septal
"""

    if HAS_FPDF:
        for name, text in [("placeholder_rcm.pdf", rcm_text), ("placeholder_arvc.pdf", arvc_text)]:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in text.strip().split("\n"):
                pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
            pdf.output(str(out / name))
            print(f"Created {out / name}")
    else:
        # Fallback: save as .txt files and note the PDF requirement
        for name, text in [("placeholder_rcm.pdf", rcm_text), ("placeholder_arvc.pdf", arvc_text)]:
            txt_path = out / name.replace(".pdf", ".txt")
            txt_path.write_text(text.strip())
            print(f"Created {txt_path} (fpdf not available, saved as .txt)")

            # Also create a minimal valid PDF manually
            pdf_path = out / name
            _create_minimal_pdf(pdf_path, text)
            print(f"Created minimal PDF: {pdf_path}")


def _create_minimal_pdf(path: Path, text: str):
    """Create a minimal valid PDF file without external dependencies."""
    # Minimal PDF structure
    lines = text.strip().split("\n")
    content_lines = []
    for line in lines:
        # Escape special PDF characters
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"BT /F1 10 Tf 72 {700 - len(content_lines) * 14} Td ({escaped}) Tj ET")

    stream_content = "\n".join(content_lines)
    stream_bytes = stream_content.encode("latin-1")

    pdf = b"%PDF-1.4\n"
    # Object 1: Catalog
    pdf += b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    # Object 2: Pages
    pdf += b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    # Object 3: Page
    pdf += b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    # Object 4: Content stream
    pdf += f"4 0 obj<</Length {len(stream_bytes)}>>stream\n".encode()
    pdf += stream_bytes
    pdf += b"\nendstream\nendobj\n"
    # Object 5: Font
    pdf += b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"

    xref_offset = len(pdf)
    pdf += b"xref\n0 6\n"
    pdf += b"0000000000 65535 f \n"
    pdf += b"0000000009 00000 n \n"
    pdf += b"0000000058 00000 n \n"
    pdf += b"0000000115 00000 n \n"
    pdf += b"0000000266 00000 n \n"
    pdf += b"0000000366 00000 n \n"
    pdf += b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n"
    pdf += f"{xref_offset}\n".encode()
    pdf += b"%%EOF"

    path.write_bytes(pdf)


if __name__ == "__main__":
    create_placeholder_pdfs()
