import subprocess

pdf_content = []
for i in range(1, 10):
    pdf_content.append(f"%PDF-1.4\n1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n3 0 obj <</Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 500 800] /Contents 5 0 R>> endobj\n4 0 obj <</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> >> >> endobj\n5 0 obj <</Length 44>> stream\nBT /F1 24 Tf 100 700 Td (Page {i} Data) Tj ET\nendstream endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000212 00000 n \n0000000293 00000 n \ntrailer <</Size 6 /Root 1 0 R>>\nstartxref\n387\n%%EOF\n")

# Normally we'd use a real library. For this script let's just make a very fast real pdf using macOS native tools if possible or python.
# Actually, since we need a valid 10 page PDF, let's use python's basic commands to generate it or download one.
