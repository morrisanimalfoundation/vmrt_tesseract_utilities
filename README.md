# Veterinary Medical Record Transcriber (VMRT) Tesseract Utilities
The Golden Retriever lifetime study has thousands of electronic medical records (EMRs) that have valuable information. The VMRT project is an attempt to automate data extraction from these EMRs. This repository contains some very simple and crude Tesseract scripts to help evaluate our dataset. The unstructured text extracted from the EMRs may or may not be valuable, but understanding the quantity of low confidence records is very useful.

Goals
* Build dataset to understand composition of EMRs
  * Kind of files
  * Enrollment status
* Determine confidence scores for ORC extraction from PDFs
* Evaluate extracted text to determine Tesseract fit for project

# Running the scripts
The scripts are easily run via the Dockerfile included in this repo.
1. Build the container like usual. `docker build -t <container name> .` Run the scripts `docker run --rm -v <path to data>/data -v <path to code>/workspace <image name> <script name>`
2. To produce a file map that is compatible with the Tesseract utilitiy run the file_info.py script over the data directory. Output is printed to stdout.
3. To process the file map produced above run the image_to_text.py with the file map. An output directory with an `unstructured_text` folder is also required. Output is dumped to output folder.
