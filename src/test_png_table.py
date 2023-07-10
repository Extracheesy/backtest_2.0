


import PyPDF2

# List of input EPS files
input_files = ['history_sum1.eps', 'history_sum2.eps', 'history_sum3.eps', 'history_sum4.eps']

# Create a new PDF object
output_pdf = PyPDF2.PdfFileWriter()

# Iterate over the input EPS files
for input_file in input_files:
    # Create a new PDF object for the current EPS file
    input_pdf = PyPDF2.PdfFileReader(input_file)

    # Get the number of pages in the current EPS file
    num_pages = input_pdf.getNumPages()

    # Iterate over the pages and add them to the output PDF
    for page_num in range(num_pages):
        page = input_pdf.getPage(page_num)
        output_pdf.addPage(page)

# Save the merged PDF as an EPS file
output_file = 'merged_output.eps'
with open(output_file, 'wb') as f:
    output_pdf.write(f)

