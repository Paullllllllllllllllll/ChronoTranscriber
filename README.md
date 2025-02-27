# ChronoTranscriber

ChronoTranscriber is a comprehensive Python-based project designed to digitize and transcribe a variety
of historical sources provided as input files—including PDFs and image folders—using multiple transcription
methods. It is particularly useful for researchers, archivists, and digital humanities professionals. By
combining native PDF extraction, Tesseract OCR, and advanced GPT-4o transcription, the system addresses
diverse document types, from searchable PDFs to scanned images, offering both synchronous processing for
individual documents and asynchronous batch processing for large-scale, cost-efficient transcription tasks.


## Overview

ChronoTranscriber is built to streamline the digitization and transcription of documents for historical
research. It offers flexible transcription options tailored to different types of documents and research needs:

- **Target Audience:**  
  Designed for researchers, archivists, digital humanities scholars, and institutions seeking to
  digitize and analyze historical documents.

- **Applications:**  
  Facilitates text extraction from archival PDFs and images, making it easier to conduct text analysis,
  content indexing, and data-driven historical research.

- **Transcription Methods:**  
  Provides multiple methods to accommodate various document types:
  - **Native PDF Extraction:** Directly extracts text from searchable PDFs, outputting results as plain text
    files.
  - **Tesseract OCR:** Uses optical character recognition to transcribe scanned PDFs and images.
  - **GPT-4o Transcription:** Leverages OpenAI’s GPT-4o model for high-quality, context-aware transcriptions,
    available in both synchronous and asynchronous batch modes.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements & Dependencies](#system-requirements--dependencies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Features](#features)
- [Workflow](#workflow)
- [Usage](#usage)
  - [PDF Processing](#pdf-processing)
  - [Image Folder Processing](#image-folder-processing)
  - [Batch Processing](#batch-processing)
- [Troubleshooting and FAQ](#troubleshooting-and-faq)
- [Contributing and Development Guidelines](#contributing-and-development-guidelines)
- [Possible Future Extension](#possible-future-extension)
- [Contributing](#contributing)
- [Contact and Support](#contact-and-support)
- [License](#license)

## Repository Structure

```
ChronoTranscriber/
├── config/
│   ├── concurrency_config.yaml
│   ├── image_processing_config.yaml
│   ├── model_config.yaml
│   └── paths_config.yaml
├── main/
│   ├── cancel_batches.py
│   ├── check_batches.py
│   └── unified_transcriber.py
├── modules/
│   ├── batching.py
│   ├── concurrency.py
│   ├── config_loader.py
│   ├── image_utils.py
│   ├── logger.py
│   ├── multiprocessing_utils.py
│   ├── openai_utils.py
│   ├── pdf_utils.py
│   ├── text_processing.py
│   ├── user_interface.py
│   └── utils.py
├── schemas/
│   └── transcription_schema.json
├── system_prompt/
│   └── system_prompt.txt
├── requirements.txt
└── README.md
```


## System Requirements & Dependencies

- **Python Version:**  
  Will work with Python 3.12 or later.

- **Further Dependencies:**
  - A full list of dependencies can be found in `requirements.txt`.


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Paullllllllllllllllll/ChronoTranscriber.git
   cd ChronoTranscriber
   ```

2. **Set Up the Environment:**
   Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Adjust PYTHONPATH:**
   To ensure that Python can find the `modules` package, update your PYTHONPATH. For example, on Linux/Mac:
   ```bash
   export PYTHONPATH="$PYTHONPATH:$(pwd)/modules"
   ```
   On Windows (PowerShell):
   ```powershell
   $env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)\modules"
   ```
   Alternatively, you can add the above commands to your shell profile or run them each time before running the project.

5. **Set Environment Variables:**
   ```bash
   export OPENAI_API_KEY=your_openai_api_key  # On Linux/Mac
   set OPENAI_API_KEY=your_openai_api_key     # On Windows
   ```

6. **Run the Transcriber:**
   ```bash
   python main/unified_transcriber.py
   ```

## Configuration

All configuration settings are managed through YAML files in the `config/` directory:

- **paths_config.yaml:**  
  Defines input/output directories, logging directory, and cleanup options.
- **model_config.yaml:**  
  Configures the transcription model (name, temperature, max tokens, etc.).
- **image_processing_config.yaml:**  
  Specifies image processing parameters such as DPI, grayscale conversion, and border removal.
- **concurrency_config.yaml:**  
  Sets concurrency limits and delays for transcription and image processing tasks.

ChronoTranscriber supports both absolute and relative file paths, improving portability across different environments and operating systems.

In the `paths_config.yaml` file, you can use these options:

```yaml
general:
  # Enable relative path support (default: false)
  allow_relative_paths: true
  
  # Base directory for resolving relative paths
  base_directory: "."
```

## Features

- **Multi-Method Transcription:**  
  - *Native Extraction:* Automatically detects and extracts text from searchable PDFs without additional
    processing.
  - *Tesseract OCR:* Converts images and scanned documents into text using established OCR techniques.
  - *GPT-4o Transcription:* Provides advanced transcription capabilities with natural language understanding,
    ideal for documents that require contextual analysis.
  - *Batch Processing:* Supports asynchronous processing of large document sets via OpenAI’s Batch API,
    enhancing efficiency and reducing costs.

- **Configurable Image Processing:**  
  - Customizable DPI settings, grayscale conversion, border removal, and transparency handling.
  - Options to adjust image resolution for optimal OCR and transcription accuracy.

- **Concurrency and Multiprocessing:**  
  - Asynchronous execution with configurable concurrency limits and task delays.
  - Multiprocessing support for intensive image processing tasks to maximize performance on multi-core systems.

- **Robust Logging and Error Handling:**  
  - Detailed logging for monitoring and troubleshooting.
  - Built-in error detection and recovery mechanisms ensure smooth processing even when issues arise.


## Workflow

1. **Configuration:**  
   - Update YAML configuration files in the `config/` directory to define file paths, model parameters,
     image processing options, and concurrency settings.
   - Configure logging paths and cleanup preferences via `paths_config.yaml`.

2. **Processing Documents:**  
   - **PDF Processing:**  
     - For native PDFs, the system extracts text directly and outputs a `.txt` file.
     - For scanned PDFs, the system extracts images and processes them using Tesseract OCR or GPT-4o
       transcription.
   - **Image Folder Processing:**  
     - Processes all images within specified folders, with options for pre-processing and transcription using
       the chosen method.
   - **Batch Processing:**  
     - Enables asynchronous transcription using GPT-4o via OpenAI’s Batch API, with scripts available for
       monitoring and cancellation of batch jobs.

3. **Output Generation:**  
   - Transcriptions are saved as `.txt` files, with JSONL records created for metadata and processing logs.
   - Temporary files and directories (such as `raw_images` and `preprocessed_images`) are managed based on
     user-defined cleanup settings.


## Usage

### PDF Processing

1. **Run the Unified Transcriber:**
   ```bash
   python main/unified_transcriber.py
   ```

2. **Select Processing Type:**  
   When prompted, choose **PDFs** by entering `2`.

3. **Choose PDF Processing Method:**  
   - For **native extraction**, the system directly extracts text from searchable PDFs.
   - For **non-native PDFs**, choose between Tesseract OCR and GPT-4o transcription. Batch processing is
     available for GPT-4o.

4. **Output:**  
   Transcribed text is saved in the designated output directory along with corresponding JSONL records.

### Image Folder Processing

1. **Run the Unified Transcriber:**
   ```bash
   python main/unified_transcriber.py
   ```

2. **Select Processing Type:**  
   Choose **Images** by entering `1`.

3. **Folder Selection:**  
   The script expects images to be organized in subfolders for better workflow coherence. However, if the image 
   input directory contains images directly (i.e., no subfolders), the script will process those images as a single folder.

4. **Choose Transcription Method:**  
   Options include GPT-4o and Tesseract OCR. Batch processing is available for GPT-4o.

5. **Output:**  
   Processed images are stored in dedicated subfolders with transcribed text saved as `.txt` files
   alongside JSONL records.

### Batch Processing

- **Checking Batches:**  
  Use `python main/check_batches.py` to review and download completed batch job outputs.

- **Cancelling Batches:**  
  Use `python main/cancel_batches.py` to cancel ongoing batch jobs that are not in a terminal state.

## Troubleshooting and FAQ

### Troubleshooting

- **OPENAI_API_KEY Not Set:**  
  Ensure the `OPENAI_API_KEY` environment variable is properly configured before using GPT-based transcription tasks.

- **Missing Configuration Files:**  
  Verify that all necessary YAML configuration files are present in the `config/` directory.

- **PDF Extraction Errors:**  
  - For native extraction, confirm that the PDF is searchable.
  - For scanned PDFs, check the image extraction and processing settings.

- **Batch Processing Issues:**  
  Check your network connectivity and review OpenAI API usage limits if batch submissions fail.

## Contributing and Development Guidelines

Contributions are welcome! When contributing:
- Contact the main developer before adding any new features.
- Follow good Python programming practices.
- Update all relevant configuration files and documentation accordingly.
- Follow the repository’s coding style and contribution guidelines.

## Possible Future Extension

Options for possible future extensions include:

- **Enhanced Error Handling:**  
  Improve error recovery mechanisms and user notifications.
- **Extended Format Support:**  
  Add support for additional file formats beyond PDFs and common image types.
- **Performance Optimizations:**  
  Enhance concurrency management and processing speeds for large-scale tasks.
- **User Interface Improvements:**  
  Develop a graphical user interface (GUI) for non-technical users.
- **Integration with Digital Archives:**  
  Enable direct integration with digital archive platforms for seamless data ingestion.

## Contributing

Contributions are welcome! When contributing:
- Contact the main developer before adding any new features.
- Follow the repository’s coding style and contribution guidelines.


## Contact and Support

- **Main Developer:**  
  For support, questions, or to discuss contributions, please open an issue on GitHub or contact via email at
  [paul.goetz@uni-goettingen.de](mailto:paul.goetz@uni-goettingen.de).

## License

This project is licensed under the MIT License.
