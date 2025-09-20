from PyPDF2 import PdfReader
reader = PdfReader("./data/resume.pdf")
text = ""
print(reader.pages[0].extract_text())
