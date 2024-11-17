# RAG (Retrieval-Augmented Generation) Project

RAG is a simple Node.js project that leverages LangChain.js and OpenAI API to process and retrieve information from documents stored in an S3 bucket. It uses vector databases such as Pinecone or Unstructured for efficient document chunking and context retrieval.

## Features

- **Document Processing:** Reads documents from an S3 location and chunks them for better storage and retrieval.
- **Vector Database Integration:** Stores document chunks in vector databases like Pinecone and Unstructured for optimized searching.
- **Context Retrieval:** Retrieves relevant context based on user prompts using Pinecone.
- **AI-Powered Responses:** Provides user outputs powered by OpenAI API, ensuring accurate and relevant answers.

## Technologies Used

- **Node.js**: Server-side JavaScript runtime.
- **LangChain.js**: For document chunking and language model chaining.
- **AWS S3**: For document storage.
- **Pinecone / Unstructured**: As vector databases for fast and scalable context retrieval.
- **OpenAI API**: For generating AI-driven responses based on the retrieved context.

## Prerequisites

- **Node.js** installed on your system.
- AWS S3 bucket with documents to process.
- Accounts and API keys for:
  - OpenAI
  - Pinecone (or Unstructured)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG.git
   cd RAG
2. Install dependencies:
   npm install
3. **Set up environment variables: Create a .env file in the project root and add the following:**
   ```bash
    AWS_ACCESS_KEY_ID=<your-aws-access-key>
    AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>
    S3_BUCKET_NAME=<your-s3-bucket-name>
    PINECONE_API_KEY=<your-pinecone-api-key>
    OPENAI_API_KEY=<your-openai-api-key>
4. Run the application: npm start
   
## Usage  
  - Upload your documents to the specified S3 bucket.
  - Start the application and provide a user prompt.
    
# The application will:
  - Fetch documents from S3.
  - Chunk them and store the chunks in the vector database.
  - Retrieve the most relevant chunks based on your prompt.
  - Use OpenAI to generate an output based on the retrieved context.
    
## Future Improvements
  - Support for additional vector databases.
  - Enhanced document processing capabilities (e.g., PDF parsing, OCR).
  - Add user interface for ease of interaction.
    
## Contributing
  Feel free to fork the repository, make improvements, and submit pull requests!
  
## License
  This project is licensed under the MIT License.
  
