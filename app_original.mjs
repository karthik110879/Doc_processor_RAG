 

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"; 
// vector store packages 
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";  
import { StringOutputParser } from "@langchain/core/output_parsers"; 
// pine cone
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
// aws s3 loader
import { S3Loader } from "langchain/document_loaders/web/s3";
// encv laoder
import * as dotenv from "dotenv"; 
  

// Load the API keys from .env
dotenv.config();  

/**Initialize Pinecone instance and OpenAI model */
// new pinecone instance
const pinecone = new Pinecone();
let pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_DEV);  
//set these from handler 
let bucket = null;
let awsLocation = null;
let docId = null;
let epicId = null;
let staging = 'prod' | 'dev'; 
var docs = null; // a doc loaded from s3 initially empty
var prompt = null;
var key = null;

//open ai instance
const model = new OpenAI({ 
    // model: "gpt-3.5-turbo", 
    temperature: 0.6,
    verbose: true
});

async function loadDocument(bucketLoc, fileKey, location) {
    new Promise(async (resolve, reject) => {

        try {

            const s3Loader = new S3Loader({
                bucket: bucketLoc,
                key: fileKey,
                s3Config: {
                    region: process.env.MY_AWS_REGION,
                    accessKeyId: process.env.MY_AWS_ACCESS_KEY_ID,
                    secretAccessKey: process.env.MY_AWS_SECRET_ACCESS_KEY,
                },
                unstructuredAPIURL: process.env.UNSTRUCTURED_API_URL,
                unstructuredAPIKey: process.env.UNSTRUCTURED_API_KEY,
            })
             
            let doc = await s3Loader.load();
            resolve(doc); 
        } catch (error) { 
            throw new Error(error);
        }
    })
}
 
 
/**
 * 
 * Utility Functions 
 */
// see if a given index is empty
async function isIndexEmpty()   {
    try { 
        // do some basic query to check if the index contains any items in it(got this from gpt)
        const result = await pineconeIndex.query({
            vector: new Array(1536).fill(0), // Example vector with zeros; adjust size as needed
            topK: 1, // will see if there is atleast one result
        });

        if (await result?.matches?.length > 0) { 
            return false; // Index is not empty
        } else { 
            return true; // Index is empty
        }
    } catch (error) { 
        throw error; // Re-throw error for further handling
    }
};

// Define a function to normalize the content of the documents
function normalizeDocuments(docs) {
    console.log('Normalizing the document');
    return docs.map((doc) => {
      if (typeof doc.pageContent === "string") {
        return doc.pageContent;
      } else if (Array.isArray(doc.pageContent)) {
        return doc.pageContent.join("\n");
      }
    });
}

// Function to summarize given text using out defined model
async function summarizeChunk(text) {
    // console.log('Summarizing the given text');
    const prompt = `Summarize the following text:\n\n${text}\n\:`;
    const response = await model.invoke(prompt);
    // console.log('Generated summary');
    return response;
}


// get the no. of tokens (use this to know how many tokens are being passwd to the llm)
async function getTokenCount(docs) {

    const embeddings = new OpenAIEmbeddings();
    // token count holder
    let tokens = 0;
    
    for(const doc of documents) {
      const tokenCount = await embeddings.countTokens(doc);
      tokens += tokenCount;
    }
    
    return tokens; 
} 



/**
 * ********** MAIN FUNCTION ************
*/
 
/**
 * @description - this functions  takes a prompt and uses a vector store and llm to return response
 * This function will do the following tasks:
 *  - Checks if an existing vector store is available in Pinecone
 *  - If an existing vector store is found, it loads it. Otherwise, it creates a new vector store from the documents dir.
 *  - Uses the vector store to perform a similarity search with the provided prompt to retrieve relevant chunks of text.
 *  - Builds a base template to prompt the llm with provided context 
 *  - Queries the LLM using the constructed prompt and returns the response.
 * @returns the response from the LLm 
 */
const processDocs = async (prompt) => {
    try {

        let vectorStore;    
        const isIndexempty = await isIndexEmpty(); 

        if(isIndexempty === false) {// access existing store for now (need to be dynamic based n te inxed)
           
            vectorStore = await PineconeStore.fromExistingIndex( 
                new OpenAIEmbeddings(),
                {
                    pineconeIndex, 
                }
            );  
            
        } else { // create new store
            
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
                keepSeparator: false
            });
         
            // normalize the document
            const normalizedDocs = normalizeDocuments(docs); 
            let splitDocs = await textSplitter.createDocuments(normalizedDocs); 
            
            // vector for better output 
            let summarizedDocs = await Promise.all(splitDocs.map(async (doc) => {
                doc.pageContent = await summarizeChunk(doc.pageContent);
                doc.metadata['epicId'] = epicId;
                doc.metadata['docId'] = docId;
                doc.id = docId;
                return doc;
            }));
         
            //Generate the vector store from the documents ==> 
            //creating a vector store using pineCone
            vectorStore = await PineconeStore.fromDocuments(
                summarizedDocs,
                new OpenAIEmbeddings(), 
                {
                    pineconeIndex,
                    maxConcurrency: 5,
                }
            ); 
            
            await new Promise(resolve => setTimeout(resolve, 6000));
        }

  
        //Query docs with filter
        const filter = {
            epicId: {"$eq":epicId}, 
            docId: {"$eq": docId}// i can also add doc id if needed
        }
        /**Search the vector DB */
        const results = await vectorStore.similaritySearch(prompt,3,filter); 

        // build a template
        const template = `
          You are a Agile Coach, assisting on creating a list of features from the given context.
          ####
          Context: {context} 
          #### 
          Question: {question} 
          ####
          Answer:
        `; 

        const customRagPrompt = PromptTemplate.fromTemplate(template); 

        const ragChain = await createStuffDocumentsChain({
            llm:model,
            prompt: customRagPrompt,
            outputParser: new StringOutputParser(),
        }); 
  
        //ask the model with the prompt and the context
        const response = await ragChain.invoke({
            question: prompt,
            context:results,
        });

        return response;

    } catch (error) {
        //  Throw error if any of the above provess fail 
        throw new Error(error) 
    }
}
 
 
// this is what lambda will use to process
export const handler = async (event) => { 
    let dummyprompt = 'Give me top 6 important features of kanban teamroom';

    // set the values here
    bucket = event.bucket;//string
    docId = event.documentId;//string
    epicId = event.epicId;//string
    staging = event.stage;//string
    key = event.key;
    awsLocation = `s3://${bucket}/${key}`;
    prompt = event.humanPrompt ? event.humanPrompt : dummyprompt;
    // prompt = event.prompt;

    pineconeIndex = staging === 'prod' ? pinecone.Index(process.env.PINECONE_INDEX_PROD) : pinecone.Index(process.env.PINECONE_INDEX_DEV); 
  
    try {
        docs = await loadDocument(bucket, key, awsLocation);
        const result = await processDocs(prompt);
        return {
            statusCode: 200,
            body: JSON.stringify({ result }),
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message }),
        };
    }
};