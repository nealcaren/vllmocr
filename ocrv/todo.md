# OCRV TODO

## Potential Improvements

- **Error Handling:**
    - Implement more robust error handling and reporting throughout the application.  Consider specific exception types for different error scenarios (e.g., API key errors, image processing errors, LLM communication errors).
    - Add retries with exponential backoff for API calls to handle transient network issues or rate limiting.

- **Configuration:**
    - Allow overriding default model choices per provider via command-line arguments.
    - Consider supporting environment variables for API keys in addition to the configuration file.
    - Validate configuration file entries on load.

- **Image Processing:**
    - Explore more advanced image preprocessing techniques (e.g., noise reduction, contrast enhancement) to improve OCR accuracy.  Offer these as configurable options.
    - Investigate handling different image orientations automatically, rather than requiring manual rotation specification.
    - Add support for more image formats.

- **LLM Interaction:**
    - Implement a caching mechanism for LLM responses to reduce API calls and improve performance.
    - Explore asynchronous API calls to improve responsiveness, especially when processing multiple images or PDFs.
    - Add more detailed logging of LLM interactions (e.g., request and response payloads, timing information).

- **Prompt Engineering:**
    - Experiment with different prompts to optimize OCR accuracy for various document types and languages.
    - Allow users to specify custom prompts via a command-line option or configuration file.

- **Testing:**
    - Expand test coverage to include more edge cases and error scenarios.
    - Add integration tests that interact with the actual LLM APIs (using mock responses where appropriate).
    - Consider performance testing to identify bottlenecks.

- **PDF Processing:**
    - Add options for handling multi-page PDFs, such as specifying page ranges or extracting text from all pages.
    - Explore alternative PDF parsing libraries for improved performance or robustness.

- **Command-Line Interface:**
    - Improve the user experience of the CLI with better argument parsing, help messages, and progress indicators.
    - Consider adding options for outputting results in different formats (e.g., JSON, plain text).

- **Extensibility:**
    - Design a plugin architecture to allow users to easily add support for new LLM providers or image processing techniques.

- **Documentation:**
    - Create comprehensive documentation, including usage examples, API reference, and troubleshooting guides.

- **Support other languages:**
    -  Allow the user to specify a language, and pass that to the LLM.

- **Batch Processing:**
    - Implement batch processing of images and PDFs to improve efficiency for large workloads.
