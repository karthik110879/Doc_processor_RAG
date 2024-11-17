

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
// vector store packages 
import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
// pine cone
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { loadSummarizationChain } from "langchain/chains"; 

// aws s3 loader
import { S3Loader } from "langchain/document_loaders/web/s3";
// encv laoder
import * as dotenv from "dotenv";


// Load the API keys from .env
dotenv.config();

// constants
const DEV_STAGE = 'dev';
const PROD_STAGE = 'prod';

// processing types
const SUMMARIZE = 'summarize'; // will pick large sample chunks and summarize the document 
const EXTRACT = 'extract'; // will pick large sample chunks and generate the features
const CHAT_BETA = 'question'; // will answer the question based on the prompt from the user (WIP)

/**Initialize Pinecone instance and OpenAI model */
// new pinecone instance
const pinecone = new Pinecone();
let pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_DEV);

//set these from handler 
let bucket = null;
let docId = null;
let epicId = null;
let staging = DEV_STAGE;
var docs = null; // a doc loaded from s3 initially empty
var prompt = null;
let key = null;
let processType = null;
// let vectorStoreRef;

//open ai instance
const model = new OpenAI({
    // model: "gpt-3.5-turbo", 
    temperature: 0.0,
    verbose: false
});
 

/**
 * Loads a document from an S3 bucket using the specified bucket location and file key.
 * 
 * This will creates an instance of `S3Loader` 
 * S3Loader will be configured with the provided S3 'bucket' and 'key', 
 * along with this we also configure the necessary AWS credentials and 'Unstructured' API configurations. 
 * then loads the document from the S3 bucket and returns it.
 * 
 * @param bucketLoc - The location (name) of the S3 bucket where the document is stored.
 * @param fileKey - The key (path or identifier) of the file with in the S3 bucket.
 * @returns promise that resolves after the document is loaded.
 * @throws An error will be thrown, if any 
 */
async function loadDocument(bucketLoc, fileKey) {

    try {
        console.log('Loading a new document from S3');
        
        const s3Loader = new S3Loader({
            bucket: bucketLoc,
            key: fileKey,
            s3Config: {
                region: isProd() ? process.env.MY_AWS_REGION : process.env.MY_AWS_REGION_DEV,
                accessKeyId: process.env.MY_AWS_ACCESS_KEY_ID,
                secretAccessKey: process.env.MY_AWS_SECRET_ACCESS_KEY,
            },
            unstructuredAPIURL: process.env.UNSTRUCTURED_API_URL,
            unstructuredAPIKey: process.env.UNSTRUCTURED_API_KEY,
        })

        let doc = await s3Loader.load();
        return doc;

    } catch (error) {
        console.log('Error while Loading a new document from S3');
        throw new Error(error);
    }
}


// Define a function to normalize the content of the documents
/**
 // normalizes the content of an array of documents.
 * This will iterates over each document and processes its `pageContent` property:
 * if `pageContent` is a string, it is returned the pageContent without any manipulation .
 * if `pageContent` is an array, the array elements are joined into a single string, 
 *   separated by newline.
 * @param {*} docs - array of documents
 * @returns 
 *        - new array containing the normalized content of each document as a string
 */
function normalizeDocuments(docs) { 
    return docs.map((doc) => {
        if (typeof doc.pageContent === "string") {
            return doc.pageContent;
        } else if (Array.isArray(doc.pageContent)) {
            return doc.pageContent.join("\n");
        }
    });
}

// Function to summarize given text using out defined model
/**
 * summarizes a given text chunk using defined model.
 * 
 * this will takes a text input, constructs a prompt to request a summary,
 * and then sends the prompt to the defined model for processing. Summarized
 * response will be returned as the output. 
 * @param text - chunk to be summarized
 * @returns this will return a promise , need to wait for the promise.
 */
async function summarizeChunk(text) { 
    const prompt = `Summarize the following text:\n\n${text}\n\:`;
    const response = await model.invoke(prompt); 
    return response;
}

/** utility function to know the enironment is dev or PROD */
function isProd() {
    return (staging === PROD_STAGE);
}


/**
 * see if a given index is empty
 * checks if a given index is empty by querying with a vector. 
 * 
 * This will queries the Pinecone index using a vector and a filter based on `epicId` and `docId`
 * It returns `true` if no matches are found(index is empty), and `false` if at least one match is found(is not empty).
 * 
 * @returns resolvs to a bool value | true = index is empty | false = not empty 
 * @throws throws error if the function fails
 */ 
async function isIndexEmpty() {
    try { 
        const filter = {
            epicId: { "$eq": epicId },
            docId: { "$eq": docId }
        }
        const result = await pineconeIndex.query({
            vector: new Array(1536).fill(0), // Example vector with zeros; adjust size as needed
            topK: 1, // will see if there is atleast one result
            filter
        }); 
        // console.log('results', result);
        
        if (result && result?.matches?.length > 0) {
            return false; // Index is not empty
        } else {
            return true; // Index is empty
        }
    } catch (error) { 
        throw error; // again throw error for further handling
    }
};

/**
 * // Function to randomly select given no of chnks from an array
 * @param {*} chunks - chunks from the existing vector(whch matches out filter)
 * @param {*} count - number of chunks to select 
 * @returns 
 */ 
async function getRandomChunks(chunks, count) { 

    const matches = await chunks.matches; 
    // shuffle the matches array 
    
    const shuffled = matches.sort(() => 0.5 - Math.random()); 
    // select the no of chunks wanted
    const selectedChunks = await shuffled.slice(0, count);  
    // return this "selectedChunks"
    return selectedChunks
}

// Assuming you have a function to get a vector from text
async function getQueryVector(text) {
    const embedding = new OpenAIEmbeddings(); // Initialize your embeddings
    const vector = await embedding.embedQuery(text); // Get vector for the text
    return vector;
}

async function selectRandomTextChunk(queryResult) {  
    // Extract all text chunks from the metadata of the query result
    const textChunks = queryResult.matches.flatMap(match => match.metadata.text); 
    
    if (textChunks.length === 0) {
        throw new Error("No text chunks found in the query result.");
    }

    // Select a random index from the array of text chunks
    const randomIndex = Math.floor(Math.random() * textChunks.length);

    // Return the randomly selected text chunk
    return textChunks[randomIndex];
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
const processDocs = async () => {
    try {
        let vectorStoreRef = null;  
        
        // clear the existing records in the pine coe index dev/prod
        if (await isIndexEmpty() === false) { 
            // load the existing here 
            vectorStoreRef = await PineconeStore.fromExistingIndex(
                new OpenAIEmbeddings(),
                {
                    pineconeIndex,
                    maxConcurrency: 5,

                }
            );
            
        } else {    
            //load the fresh document into the store from s3
            docs = await loadDocument(bucket, key);
            if(docs) { 
                // textSplitter : Used to split text from the loaded document 
                //this will chunk the text into 1000 with 300 overlap
                const textSplitter = new RecursiveCharacterTextSplitter({
                    chunkSize: 500,
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
     
                //creating a vector store using pineCone
                vectorStoreRef = await PineconeStore.fromDocuments(
                    summarizedDocs,
                    new OpenAIEmbeddings(),
                    {
                        pineconeIndex, 
                    }
                ); 
                await new Promise(resolve => setTimeout(resolve, 6000));
            }
        }
 
        if (processType === EXTRACT) {  
            console.log('Extracting the document');
            
            // build a prompt template
            const template = ` 
                ####
                Context: {context} 
                #### 
                Question: {question} 
                ####
                Provide a clear, prioritized list of the top features to focus on, with a brief explanation for each. 
                Limit the list to between 5 and 8 features. If fewer than 5 are found, include additional relevant 
                features to reach this minimum. Separate the feature title and feature description by a ":".
            `;

            // Retrieve all matching documents based on the filter\
            const filter = {
                epicId: { "$eq": epicId },
                docId: { "$eq": docId }
            }
           
            // Step 2: Query Pinecone to retrieve relevant documents
            const relevantDocs = await pineconeIndex.query({
                vector: Array(1536).fill(0), // Use a dummy vector to retrieve all relevant documents
                filter: filter,
                topK: 60, // Retrieve a large number to ensure a good sample from the store
                includeMetadata: true
            }); 
            
            const randomTextChunk = await selectRandomTextChunk(relevantDocs);
            const queryResult = await pineconeIndex.query({
                vector: await getQueryVector(randomTextChunk),
                filter: filter,
                topK: 60, // using 100 because i wana get goo number of chunks to select from 
                includeMetadata: true
            });
            //Will get random chunks here
            const randomChunks = await getRandomChunks(queryResult, 50);  
              
            const customRagPrompt = PromptTemplate.fromTemplate(template);
            const ragChain = await createStuffDocumentsChain({
                llm: model,
                prompt: customRagPrompt,
                outputParser: new StringOutputParser(),
            });

            // Define the question and context
            const question = "Give me a list of features from the given context?";   

            // Prepare the context as an array of documents 
            const context = randomChunks.map(chunk => ({
                pageContent: chunk.metadata.text,
                id: chunk.id,
                score: chunk.score
            })); 
        
            // const txtContext = context.map(({ pageContent: chunk.metadata.text }));
            // console.log('txtContext', txtContext);

            // Invoke the chain with the context and question
            const response = await ragChain.invoke({
                context: context,
                question: question
            });
            
            return response;

        }

        if(processType === SUMMARIZE){  
            console.log('Summarizing the document');
            
            // build a template
            const template = `
            You are an expert in summarizing complex documents. Your task is to create a clear, 
            concise, and informative summary from the given context. 
         
            Guidelines:
            1. The summary should be brief, with a minimum of 50 lines and maximum of 70 lines.
            2. Focus on the key points, major themes, and critical information.
            3. Avoid unnecessary details or repetition.

            ####
            Context: {context} 
            #### 
            Question: {question} 
            ####
            Summary:
            `;
            
            // Retrieve all matching documents based on the filter\
            const filter = {
                epicId: { "$eq": epicId },
                docId: { "$eq": docId }
            }
            // const dummyVector = Array(1536).fill(0); 

            // Step 2: Query Pinecone to retrieve relevant documents
            const relevantDocs = await pineconeIndex.query({
                vector: Array(1536).fill(0), // Use a dummy vector to retrieve all relevant documents
                filter: filter,
                topK: 15, // Retrieve a large number to ensure a good sample from the store
                includeMetadata: true
            });
            const randomTextChunk = await selectRandomTextChunk(relevantDocs);
            const queryResult = await pineconeIndex.query({
                vector: await getQueryVector(randomTextChunk),
                filter: filter,
                topK: 60, // retrieve a large number to ensure a good sample from the store
                includeMetadata: true
            }); 

            // Select 5-6 random chunks from the results -> changed to 20 based on test results
            const randomChunks = await getRandomChunks(queryResult, 50).catch(error => console.log(error));
            
            const customRagPrompt = PromptTemplate.fromTemplate(template);
            const ragChain = await createStuffDocumentsChain({
                llm: model,
                prompt: customRagPrompt,
                outputParser: new StringOutputParser(),
            });

            // Define the question and context
            const question = "Provide a brief summary of the key points and insights from the provided context.";  
            // Prepare the context as an array of documents
            const context = await randomChunks.map(chunk => ({ pageContent: chunk.metadata.text }));

            // Invoke the chain with the context and question
            const response = await ragChain.invoke({
                context: context,
                question: question
            });
            
            return response;
        } 

        if(processType === CHAT_BETA) {
            const filter = {
                epicId: {"$eq":epicId}, 
                docId: {"$eq": docId}// i can also add doc id if needed
            }

            /**Search the vector DB */
            const results = vectorStoreRef.similaritySearch(prompt,6,filter)
            // build a template
            const template = `
            You are a Q and A assistant, answering to the user from the given context.
            if you cannot generate a answer , respond with "I dont know, ask me something else".
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
        }
       
    } catch (error) {
        //  Throw error if any of the above provess fail 
        throw new Error(error)
    }
}

function clearMeta() {
    bucket = null;//string
    docId = null;//string
    epicId = null;//string
    staging = DEV_STAGE;//string
    key = null; // file name and path
    prompt = null; // optional
    processType = EXTRACT // 'e
}

// this is what lambda will use to process
export const handler = async (event) => {

    try { 
        console.log('INCOMING EVENT', event)
        clearMeta();
        // set the values here
        bucket = event.bucket;//string
        docId = event.documentId;//string
        epicId = event.epicId;//string
        staging = event.stage || DEV_STAGE;//string
        key = event.key; // file name and path
        prompt = event.humanPrompt; // optional
        processType = event.type || EXTRACT // 'extract' | 'summarize' 
        pineconeIndex = isProd() 
                        ? pinecone.Index(process.env.PINECONE_INDEX_PROD) 
                        : pinecone.Index(process.env.PINECONE_INDEX_DEV);

        const result = await processDocs();

        return {
            statusCode: 200,
            body: JSON.stringify({result})
        };

    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message })
        };
    }
};
 